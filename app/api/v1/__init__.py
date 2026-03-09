"""API v1路由初始化

此模块导出所有v1版本的路由。
"""
from app.api.v1.auth import router as auth_router
from app.api.v1.admin import router as admin_router
from app.api.v1.finance import router as finance_router
from app.api.v1.weather import router as weather_router
from app.api.v1.fortune import router as fortune_router
from app.api.v1.consumption import router as consumption_router
from app.api.v1.crawler import router as crawler_router

from fastapi import APIRouter

router = APIRouter()

router.include_router(auth_router, prefix="/auth", tags=["认证"])
router.include_router(admin_router, prefix="/admin", tags=["系统管理"])
router.include_router(finance_router, prefix="/finance", tags=["金融分析"])
router.include_router(weather_router, prefix="/weather", tags=["气象分析"])
router.include_router(fortune_router, prefix="/fortune", tags=["看相算命"])
router.include_router(consumption_router, prefix="/consumption", tags=["消费分析"])
router.include_router(crawler_router, prefix="/crawler", tags=["爬虫采集"])
