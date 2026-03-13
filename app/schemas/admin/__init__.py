"""系统管理 Schema

此模块整合所有系统管理相关的 Schema。
"""
from app.schemas.admin.user import (
    UserBase,
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse
)

from app.schemas.admin.role import (
    RoleBase,
    RoleCreate,
    RoleUpdate,
    RoleResponse
)

from app.schemas.admin.permission import (
    PermissionBase,
    PermissionCreate,
    PermissionUpdate,
    PermissionResponse
)

from app.schemas.admin.system_config import (
    SystemConfigBase,
    SystemConfigCreate,
    SystemConfigUpdate,
    SystemConfigResponse
)

from app.schemas.admin.system_log import (
    SystemLogBase,
    SystemLogCreate,
    SystemLogResponse
)

# 别名，用于向后兼容
OperationLogBase = SystemLogBase
OperationLogCreate = SystemLogCreate
OperationLogResponse = SystemLogResponse

from app.schemas.admin.dashboard import (
    DashboardOverview,
    UserStatistics,
    LogStatistics,
    SystemHealth,
    ModuleStatistic,
    RecentActivity
)

__all__ = [
    'UserBase',
    'UserCreate',
    'UserUpdate',
    'UserResponse',
    'UserListResponse',
    'RoleBase',
    'RoleCreate',
    'RoleUpdate',
    'RoleResponse',
    'PermissionBase',
    'PermissionCreate',
    'PermissionUpdate',
    'PermissionResponse',
    'SystemConfigBase',
    'SystemConfigCreate',
    'SystemConfigUpdate',
    'SystemConfigResponse',
    'SystemLogBase',
    'SystemLogCreate',
    'SystemLogResponse',
    'OperationLogBase',
    'OperationLogCreate',
    'OperationLogResponse',
    'DashboardOverview',
    'UserStatistics',
    'LogStatistics',
    'SystemHealth',
    'ModuleStatistic',
    'RecentActivity',
]
