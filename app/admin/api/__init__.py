"""系统管理 API 路由

此模块整合所有系统管理相关的 API 接口。
"""
from fastapi import APIRouter

from app.admin.api import users

router = APIRouter(prefix="/admin", tags=["系统管理"])

# 注册用户管理路由
router.include_router(users.router)

# 后续添加角色、权限等路由
# router.include_router(roles.router)
# router.include_router(permissions.router)
# router.include_router(configs.router)
# router.include_router(logs.router)
# router.include_router(dashboard.router)
