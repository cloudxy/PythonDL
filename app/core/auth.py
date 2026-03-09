"""认证和授权增强模块

此模块提供JWT令牌认证、刷新令牌、角色权限检查等高级功能。
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from sqlalchemy.orm import Session
import json
import time

from app.core.config import config
from app.core.database import get_db
from app.core.logger import get_logger
from app.core.cache import cache_manager
from app.core.security import verify_password, get_password_hash, validate_password_strength
from app.services.admin.user_service import UserService
from app.models.admin.user import User
from app.models.admin.role import Role
from app.models.admin.permission import Permission
from app.models.admin.role_permission import RolePermission

logger = get_logger("auth")

# JWT配置
import os

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", config.get("security.jwt_secret_key", "your-secret-key-change-in-production"))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", config.get("security.access_token_expire_minutes", 30)))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", config.get("security.refresh_token_expire_days", 7)))

# OAuth2密码流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间增量
        
    Returns:
        str: JWT访问令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """创建刷新令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间增量
        
    Returns:
        str: JWT刷新令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """验证JWT令牌
    
    Args:
        token: JWT令牌
        
    Returns:
        Optional[Dict[str, Any]]: 令牌数据，如果无效则返回None
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def is_token_revoked(token: str) -> bool:
    """检查令牌是否已被吊销
    
    Args:
        token: JWT令牌
        
    Returns:
        bool: 是否已被吊销
    """
    token_key = f"revoked_token:{token}"
    return cache_manager.exists(token_key)


def revoke_token(token: str) -> None:
    """吊销令牌
    
    Args:
        token: JWT令牌
    """
    token_key = f"revoked_token:{token}"
    # 获取令牌过期时间
    try:
        payload = verify_token(token)
        if payload and "exp" in payload:
            expire_time = payload["exp"] - int(time.time())
            if expire_time > 0:
                cache_manager.set(token_key, "1", expire=expire_time)
            else:
                # 令牌已过期，默认缓存24小时
                cache_manager.set(token_key, "1", expire=86400)
        else:
            # 默认缓存24小时
            cache_manager.set(token_key, "1", expire=86400)
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        # 即使出错，也要设置缓存
        cache_manager.set(token_key, "1", expire=86400)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """获取当前认证用户（OAuth2密码流版本）
    
    Args:
        token: JWT令牌
        db: 数据库会话
        
    Returns:
        User: 认证用户对象
        
    Raises:
        HTTPException: 认证失败时抛出
    """
    logger.debug(f"Authentication attempt with token: {token[:10]}...")
    
    # 检查令牌是否被吊销
    if is_token_revoked(token):
        logger.warning(f"Authentication failed: Token revoked: {token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证JWT令牌
    payload = verify_token(token)
    if not payload:
        logger.warning(f"Authentication failed: Invalid token: {token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 检查令牌类型
    token_type = payload.get("type")
    if token_type != "access":
        logger.warning(f"Authentication failed: Wrong token type: {token_type}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 获取用户信息
    user_id = payload.get("sub")
    if not user_id:
        logger.warning("Authentication failed: No user ID in token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_service = UserService(db)
    user = user_service.get_user(int(user_id))
    
    if not user:
        logger.warning(f"Authentication failed: User not found for ID: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 检查用户状态
    if not user.is_active:
        logger.warning(f"Authentication failed: User inactive: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    logger.info(f"Authentication successful for user: {user.username}")
    return user


async def get_current_user_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前认证用户（HTTP Bearer版本）
    
    Args:
        credentials: HTTP Bearer 凭证
        db: 数据库会话
        
    Returns:
        User: 认证用户对象
        
    Raises:
        HTTPException: 认证失败时抛出
    """
    if credentials is None:
        logger.warning("Authentication failed: No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    return await get_current_user(token, db)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """获取当前活跃用户
    
    Args:
        current_user: 当前认证用户
        
    Returns:
        User: 当前活跃用户
    """
    return current_user


def _get_user_role_name(user: User, db: Session) -> Optional[str]:
    """获取用户角色名称
    
    Args:
        user: 用户对象
        db: 数据库会话
        
    Returns:
        Optional[str]: 角色名称，如果没有则返回None
    """
    if hasattr(user, 'role') and user.role and hasattr(user.role, 'role_name'):
        return user.role.role_name
    elif hasattr(user, 'role_id') and user.role_id:
        # 通过role_id查询角色
        role = db.query(Role).filter(Role.id == user.role_id).first()
        if role:
            return role.role_name
    return None


def require_role(role_name: str):
    """要求特定角色的依赖工厂
    
    Args:
        role_name: 角色名称
        
    Returns:
        依赖函数
    """
    async def role_dependency(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ) -> User:
        """角色检查依赖
        
        Args:
            current_user: 当前认证用户
            db: 数据库会话
            
        Returns:
            User: 有权限的用户
            
        Raises:
            HTTPException: 无权限时抛出
        """
        # 获取用户角色名称
        user_role_name = _get_user_role_name(current_user, db)
        
        # 检查用户角色
        if user_role_name != role_name:
            logger.warning(f"Role permission denied for user: {current_user.username}, required role: {role_name}, user role: {user_role_name}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_name}' required",
            )
        
        return current_user
    
    return role_dependency


def require_permission(permission_name: str):
    """要求特定权限的依赖工厂
    
    Args:
        permission_name: 权限名称
        
    Returns:
        依赖函数
    """
    async def permission_dependency(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ) -> User:
        """权限检查依赖
        
        Args:
            current_user: 当前认证用户
            db: 数据库会话
            
        Returns:
            User: 有权限的用户
            
        Raises:
            HTTPException: 无权限时抛出
        """
        # 检查用户是否拥有指定权限
        has_permission = check_user_permission(db, current_user, permission_name)
        
        if not has_permission:
            logger.warning(f"Permission denied for user: {current_user.username}, required permission: {permission_name}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission_name}' required",
            )
        
        return current_user
    
    return permission_dependency


def require_role_bearer(role_name: str):
    """要求特定角色的依赖工厂（使用HTTP Bearer认证）
    
    Args:
        role_name: 角色名称
        
    Returns:
        依赖函数
    """
    async def role_dependency(
        current_user: User = Depends(get_current_user_bearer),
        db: Session = Depends(get_db)
    ) -> User:
        """角色检查依赖
        
        Args:
            current_user: 当前认证用户
            db: 数据库会话
            
        Returns:
            User: 有权限的用户
            
        Raises:
            HTTPException: 无权限时抛出
        """
        # 获取用户角色名称
        user_role_name = _get_user_role_name(current_user, db)
        
        # 检查用户角色
        if user_role_name != role_name:
            logger.warning(f"Role permission denied for user: {current_user.username}, required role: {role_name}, user role: {user_role_name}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_name}' required",
            )
        
        return current_user
    
    return role_dependency


def require_permission_bearer(permission_name: str):
    """要求特定权限的依赖工厂（使用HTTP Bearer认证）
    
    Args:
        permission_name: 权限名称
        
    Returns:
        依赖函数
    """
    async def permission_dependency(
        current_user: User = Depends(get_current_user_bearer),
        db: Session = Depends(get_db)
    ) -> User:
        """权限检查依赖
        
        Args:
            current_user: 当前认证用户
            db: 数据库会话
            
        Returns:
            User: 有权限的用户
            
        Raises:
            HTTPException: 无权限时抛出
        """
        # 检查用户是否拥有指定权限
        has_permission = check_user_permission(db, current_user, permission_name)
        
        if not has_permission:
            logger.warning(f"Permission denied for user: {current_user.username}, required permission: {permission_name}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission_name}' required",
            )
        
        return current_user
    
    return permission_dependency


def check_user_permission(db: Session, user: User, permission_name: str) -> bool:
    """检查用户是否拥有指定权限
    
    Args:
        db: 数据库会话
        user: 用户对象
        permission_name: 权限名称
        
    Returns:
        bool: 是否拥有权限
    """
    # 生成缓存键
    cache_key = f"user_permission:{user.id}:{permission_name}"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key)
    if cached_result is not None:
        return cached_result == "1"
    
    # 如果用户有角色，检查是否是admin
    if hasattr(user, 'role') and user.role and hasattr(user.role, 'role_name') and user.role.role_name == "admin":
        # 缓存结果
        cache_manager.set(cache_key, "1", expire=3600)  # 缓存1小时
        return True
    
    # 或者检查role_id是否为1（admin角色）
    if hasattr(user, 'role_id') and user.role_id == 1:
        # 缓存结果
        cache_manager.set(cache_key, "1", expire=3600)  # 缓存1小时
        return True
    
    # 查询用户角色对应的权限
    try:
        # 首先获取角色ID
        role_id = None
        if hasattr(user, 'role') and user.role and hasattr(user.role, 'id'):
            role_id = user.role.id
        elif hasattr(user, 'role_id') and user.role_id:
            role_id = user.role_id
        
        if not role_id:
            # 缓存结果
            cache_manager.set(cache_key, "0", expire=3600)  # 缓存1小时
            return False
        
        # 查询角色拥有的权限
        permission = db.query(Permission).filter(Permission.permission_name == permission_name).first()
        if not permission:
            # 缓存结果
            cache_manager.set(cache_key, "0", expire=3600)  # 缓存1小时
            return False
        
        # 检查角色权限关联
        role_permission = db.query(RolePermission).filter(
            RolePermission.role_id == role_id,
            RolePermission.permission_id == permission.id
        ).first()
        
        result = role_permission is not None
        # 缓存结果
        cache_manager.set(cache_key, "1" if result else "0", expire=3600)  # 缓存1小时
        return result
    except Exception as e:
        logger.error(f"Error checking permission: {str(e)}", exc_info=True)
        return False


def get_user_permissions(db: Session, user: User) -> List[str]:
    """获取用户所有权限
    
    Args:
        db: 数据库会话
        user: 用户对象
        
    Returns:
        List[str]: 权限名称列表
    """
    # 生成缓存键
    cache_key = f"user_permissions:{user.id}"
    
    # 尝试从缓存获取
    cached_result = cache_manager.get(cache_key)
    if cached_result is not None:
        try:
            return eval(cached_result)
        except:
            pass
    
    permissions = []
    
    # 首先获取用户角色
    role = None
    if hasattr(user, 'role') and user.role:
        role = user.role
    elif hasattr(user, 'role_id') and user.role_id:
        role = db.query(Role).filter(Role.id == user.role_id).first()
    
    # 如果用户角色是admin，返回所有权限
    if role and hasattr(role, 'role_name') and role.role_name == "admin":
        all_permissions = db.query(Permission).all()
        permissions = [p.permission_name for p in all_permissions]
        # 缓存结果
        cache_manager.set(cache_key, str(permissions), expire=3600)  # 缓存1小时
        return permissions
    
    # 如果没有角色，返回空列表
    if not role:
        # 缓存结果
        cache_manager.set(cache_key, str(permissions), expire=3600)  # 缓存1小时
        return permissions
    
    try:
        # 查询角色拥有的权限
        role_permissions = db.query(RolePermission).filter(RolePermission.role_id == role.id).all()
        
        # 获取权限名称
        for rp in role_permissions:
            permission = db.query(Permission).filter(Permission.id == rp.permission_id).first()
            if permission:
                permissions.append(permission.permission_name)
        
        # 缓存结果
        cache_manager.set(cache_key, str(permissions), expire=3600)  # 缓存1小时
    except Exception as e:
        logger.error(f"Error getting user permissions: {str(e)}", exc_info=True)
    
    return permissions


# 兼容旧版本认证函数（可选的向后兼容）
async def get_current_user_legacy(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """旧版本获取当前认证用户（基于会话缓存）
    
    此函数用于向后兼容，新代码建议使用get_current_user（JWT版本）
    """
    from app.core.deps import get_current_user as legacy_get_current_user
    return await legacy_get_current_user(credentials, db)