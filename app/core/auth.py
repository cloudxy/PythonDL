"""认证和授权增强模块

此模块提供 JWT 令牌认证、刷新令牌、角色权限检查等高级功能。
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    OAuth2PasswordBearer
)
from sqlalchemy.orm import Session
import time
import os

from app.core.config import config
from app.core.database import get_db
from app.core.logger import get_logger
from app.core.cache import cache_manager
from app.services.admin.user_service import UserService
from app.models.admin.user import User
from app.models.admin.role import Role
from app.models.admin.permission import Permission
from app.models.admin.role_permission import RolePermission

logger = get_logger("auth")

# JWT 配置
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    config.get(
        "security.jwt_secret_key",
        "your-secret-key-change-in-production"
    )
)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        config.get("security.access_token_expire_minutes", 30)
    )
)
REFRESH_TOKEN_EXPIRE_DAYS = int(
    os.getenv(
        "REFRESH_TOKEN_EXPIRE_DAYS",
        config.get("security.refresh_token_expire_days", 7)
    )
)

# OAuth2 密码流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建访问令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间增量
        
    Returns:
        str: JWT 访问令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )
    return encoded_jwt  # type: ignore


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建刷新令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间增量
        
    Returns:
        str: JWT 刷新令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )
    return encoded_jwt  # type: ignore


def create_password_reset_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建密码重置令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间增量，默认 1 小时
        
    Returns:
        str: JWT 密码重置令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
    
    to_encode.update({"exp": expire, "type": "password_reset"})
    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )
    return encoded_jwt  # type: ignore


def verify_password_reset_token(
    token: str
) -> Optional[Dict[str, Any]]:
    """验证密码重置令牌
    
    Args:
        token: JWT 令牌
        
    Returns:
        Optional[Dict[str, Any]]: 令牌数据，如果无效则返回 None
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        if payload.get("type") != "password_reset":
            return None
        return payload  # type: ignore
    except JWTError:
        return None


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """验证 JWT 令牌
    
    Args:
        token: JWT 令牌
        
    Returns:
        Optional[Dict[str, Any]]: 令牌数据，如果无效则返回 None
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        return payload  # type: ignore
    except JWTError:
        return None


def is_token_revoked(token: str) -> bool:
    """检查令牌是否已被吊销
    
    Args:
        token: JWT 令牌
        
    Returns:
        bool: 是否已被吊销
    """
    token_key = f"revoked_token:{token}"
    return cache_manager.exists(token_key)


def revoke_token(token: str) -> None:
    """吊销令牌
    
    Args:
        token: JWT 令牌
    """
    token_key = f"revoked_token:{token}"
    try:
        payload = verify_token(token)
        if payload and "exp" in payload:
            expire_time = payload["exp"] - int(time.time())
            if expire_time > 0:
                cache_manager.set(token_key, "1", expire=expire_time)
            else:
                cache_manager.set(token_key, "1", expire=86400)
        else:
            cache_manager.set(token_key, "1", expire=86400)
    except Exception as e:
        logger.error("Error revoking token: %s", str(e))
        cache_manager.set(token_key, "1", expire=86400)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """获取当前认证用户（OAuth2 密码流版本）
    
    Args:
        token: JWT 令牌
        db: 数据库会话
        
    Returns:
        User: 认证用户对象
        
    Raises:
        HTTPException: 认证失败时抛出
    """
    logger.debug("Authentication attempt with token: %s...", token[:10])
    
    if is_token_revoked(token):
        logger.warning("Authentication failed: Token revoked: %s...", token[:10])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_token(token)
    if not payload:
        logger.warning("Authentication failed: Invalid token: %s...", token[:10])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_type = payload.get("type")
    if token_type != "access":
        logger.warning("Authentication failed: Wrong token type: %s", token_type)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
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
        logger.warning(
            "Authentication failed: User not found for ID: %s",
            user_id
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        logger.warning("Authentication failed: User inactive: %s", user.username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    logger.info("Authentication successful for user: %s", user.username)
    return user


async def get_current_user_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前认证用户（HTTP Bearer 版本）
    
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


def _get_user_role_name(
    user: User,
    db: Session
) -> Optional[str]:
    """获取用户角色名称
    
    Args:
        user: 用户对象
        db: 数据库会话
        
    Returns:
        Optional[str]: 角色名称，如果没有则返回 None
    """
    if hasattr(user, 'role') and user.role:
        role_obj = user.role
        if hasattr(role_obj, 'role_name'):
            role_name = role_obj.role_name
            if isinstance(role_name, str):
                return role_name
    elif hasattr(user, 'role_id') and user.role_id:
        role = db.query(Role).filter(Role.id == user.role_id).first()
        if role:
            return role.role_name  # type: ignore
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
        user_role_name = _get_user_role_name(current_user, db)
        
        if user_role_name != role_name:
            logger.warning(
                "Role permission denied for user: %s, "
                "required role: %s, user role: %s",
                current_user.username,
                role_name,
                user_role_name
            )
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
        has_permission = check_user_permission(
            db,
            current_user,
            permission_name
        )
        
        if not has_permission:
            logger.warning(
                "Permission denied for user: %s, required permission: %s",
                current_user.username,
                permission_name
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission_name}' required",
            )
        
        return current_user
    
    return permission_dependency


def require_role_bearer(role_name: str):
    """要求特定角色的依赖工厂（使用 HTTP Bearer 认证）
    
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
        user_role_name = _get_user_role_name(current_user, db)
        
        if user_role_name != role_name:
            logger.warning(
                "Role permission denied for user: %s, "
                "required role: %s, user role: %s",
                current_user.username,
                role_name,
                user_role_name
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_name}' required",
            )
        
        return current_user
    
    return role_dependency


def require_permission_bearer(permission_name: str):
    """要求特定权限的依赖工厂（使用 HTTP Bearer 认证）
    
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
        has_permission = check_user_permission(
            db,
            current_user,
            permission_name
        )
        
        if not has_permission:
            logger.warning(
                "Permission denied for user: %s, required permission: %s",
                current_user.username,
                permission_name
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission_name}' required",
            )
        
        return current_user
    
    return permission_dependency


def check_user_permission(
    db: Session,
    user: User,
    permission_name: str
) -> bool:
    """检查用户是否拥有指定权限
    
    Args:
        db: 数据库会话
        user: 用户对象
        permission_name: 权限名称
        
    Returns:
        bool: 是否拥有权限
    """
    cache_key = f"user_permission:{user.id}:{permission_name}"
    
    cached_result = cache_manager.get(cache_key)
    if cached_result is not None:
        return cached_result == "1"
    
    if hasattr(user, 'role') and user.role:
        role_obj = user.role
        if hasattr(role_obj, 'role_name'):
            if role_obj.role_name == "admin":
                cache_manager.set(cache_key, "1", expire=3600)
                return True
    
    if hasattr(user, 'role_id') and user.role_id == 1:
        cache_manager.set(cache_key, "1", expire=3600)
        return True
    
    try:
        role_id = None
        if hasattr(user, 'role') and user.role:
            role_obj = user.role
            if hasattr(role_obj, 'id'):
                role_id = role_obj.id
        elif hasattr(user, 'role_id') and user.role_id:
            role_id = user.role_id
        
        if not role_id:
            cache_manager.set(cache_key, "0", expire=3600)
            return False
        
        permission = db.query(Permission).filter(
            Permission.permission_name == permission_name
        ).first()
        if not permission:
            cache_manager.set(cache_key, "0", expire=3600)
            return False
        
        role_permission = db.query(RolePermission).filter(
            RolePermission.role_id == role_id,
            RolePermission.permission_id == permission.id
        ).first()
        
        result = role_permission is not None
        cache_manager.set(
            cache_key,
            "1" if result else "0",
            expire=3600
        )
        return result
    except Exception as e:
        logger.error("Error checking permission: %s", str(e), exc_info=True)
        return False


def get_user_permissions(
    db: Session,
    user: User
) -> List[str]:
    """获取用户所有权限
    
    Args:
        db: 数据库会话
        user: 用户对象
        
    Returns:
        List[str]: 权限名称列表
    """
    cache_key = f"user_permissions:{user.id}"
    
    cached_result = cache_manager.get(cache_key)
    if cached_result is not None:
        try:
            result = eval(cached_result)
            if isinstance(result, list):
                return result
        except Exception:
            pass
    
    permissions: List[str] = []
    
    role = None
    if hasattr(user, 'role') and user.role:
        role = user.role
    elif hasattr(user, 'role_id') and user.role_id:
        role = db.query(Role).filter(Role.id == user.role_id).first()
    
    if role and hasattr(role, 'role_name'):
        if role.role_name == "admin":
            all_permissions = db.query(Permission).all()
            permissions = [
                p.permission_name
                for p in all_permissions
                if hasattr(p, 'permission_name')
            ]
            cache_manager.set(cache_key, str(permissions), expire=3600)
            return permissions
    
    if not role:
        cache_manager.set(cache_key, str(permissions), expire=3600)
        return permissions
    
    try:
        role_permissions = db.query(RolePermission).filter(
            RolePermission.role_id == role.id
        ).all()
        
        for rp in role_permissions:
            permission = db.query(Permission).filter(
                Permission.id == rp.permission_id
            ).first()
            if permission and hasattr(permission, 'permission_name'):
                perm_name = permission.permission_name
                if isinstance(perm_name, str):
                    permissions.append(perm_name)
        
        cache_manager.set(cache_key, str(permissions), expire=3600)
    except Exception as e:
        logger.error("Error getting user permissions: %s", str(e), exc_info=True)
    
    return permissions


async def get_current_user_legacy(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """旧版本获取当前认证用户（基于会话缓存）
    
    此函数用于向后兼容，新代码建议使用 get_current_user（JWT 版本）
    """
    return await get_current_user_bearer(credentials, db)
