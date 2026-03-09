"""系统管理API路由

此模块定义系统管理相关的API接口。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.schemas.admin import (
    UserCreate,
    UserUpdate,
    UserResponse,
    RoleCreate,
    RoleUpdate,
    RoleResponse,
    PermissionCreate,
    PermissionResponse,
    SystemConfigCreate,
    SystemConfigUpdate,
    SystemConfigResponse,
    OperationLogResponse
)
from app.services.admin.user_service import UserService
from app.services.admin.role_service import RoleService
from app.services.admin.permission_service import PermissionService
from app.services.admin.system_config_service import SystemConfigService
from app.services.admin.operation_log_service import OperationLogService

router = APIRouter()


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    username: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户列表"""
    user_service = UserService(db)
    users = user_service.get_users(skip=skip, limit=limit, username=username)
    return [UserResponse.from_orm(user) for user in users]


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("user:create"))
):
    """创建用户"""
    user_service = UserService(db)
    user = user_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        real_name=user_data.real_name,
        role_id=user_data.role_id
    )
    return UserResponse.from_orm(user)


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户详情"""
    user_service = UserService(db)
    user = user_service.get_user(user_id)
    if not user:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="用户不存在")
    return UserResponse.from_orm(user)


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("user:update"))
):
    """更新用户"""
    user_service = UserService(db)
    user = user_service.update_user(user_id, user_data.dict(exclude_unset=True))
    return UserResponse.from_orm(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("user:delete"))
):
    """删除用户"""
    user_service = UserService(db)
    user_service.delete_user(user_id)
    return {"message": "删除成功"}


@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取角色列表"""
    role_service = RoleService(db)
    roles = role_service.get_roles(skip=skip, limit=limit)
    return [RoleResponse.from_orm(role) for role in roles]


@router.post("/roles", response_model=RoleResponse)
async def create_role(
    role_data: RoleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("role:create"))
):
    """创建角色"""
    role_service = RoleService(db)
    role = role_service.create_role(role_data.dict())
    return RoleResponse.from_orm(role)


@router.get("/roles/{role_id}", response_model=RoleResponse)
async def get_role(
    role_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取角色详情"""
    role_service = RoleService(db)
    role = role_service.get_role(role_id)
    if not role:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="角色不存在")
    return RoleResponse.from_orm(role)


@router.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: int,
    role_data: RoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("role:update"))
):
    """更新角色"""
    role_service = RoleService(db)
    role = role_service.update_role(role_id, role_data.dict(exclude_unset=True))
    return RoleResponse.from_orm(role)


@router.delete("/roles/{role_id}")
async def delete_role(
    role_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("role:delete"))
):
    """删除角色"""
    role_service = RoleService(db)
    role_service.delete_role(role_id)
    return {"message": "删除成功"}


@router.get("/permissions", response_model=List[PermissionResponse])
async def list_permissions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取权限列表"""
    permission_service = PermissionService(db)
    permissions = permission_service.get_permissions(skip=skip, limit=limit)
    return [PermissionResponse.from_orm(p) for p in permissions]


@router.post("/permissions", response_model=PermissionResponse)
async def create_permission(
    permission_data: PermissionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permission:create"))
):
    """创建权限"""
    permission_service = PermissionService(db)
    permission = permission_service.create_permission(permission_data.dict())
    return PermissionResponse.from_orm(permission)


@router.get("/configs", response_model=List[SystemConfigResponse])
async def list_configs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取系统配置列表"""
    config_service = SystemConfigService(db)
    configs = config_service.get_configs(skip=skip, limit=limit, category=category)
    return [SystemConfigResponse.from_orm(c) for c in configs]


@router.post("/configs", response_model=SystemConfigResponse)
async def create_config(
    config_data: SystemConfigCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("config:create"))
):
    """创建系统配置"""
    config_service = SystemConfigService(db)
    config = config_service.create_config(config_data.dict())
    return SystemConfigResponse.from_orm(config)


@router.get("/logs", response_model=List[OperationLogResponse])
async def list_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = None,
    operation_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("log:view"))
):
    """获取操作日志列表"""
    log_service = OperationLogService(db)
    logs = log_service.get_logs(
        skip=skip,
        limit=limit,
        user_id=user_id,
        operation_type=operation_type
    )
    return [OperationLogResponse.from_orm(log) for log in logs]


@router.get("/dashboard")
async def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取仪表盘数据"""
    user_service = UserService(db)
    log_service = OperationLogService(db)
    
    return {
        "user_count": user_service.count_users(),
        "active_users": user_service.count_active_users(),
        "today_logs": log_service.count_today_logs(),
        "recent_logs": log_service.get_recent_logs(limit=10)
    }
