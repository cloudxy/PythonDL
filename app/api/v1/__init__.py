"""API v1 路由初始化

此模块导出所有 v1 版本的路由。
"""
from app.api.v1.auth import router as auth_router
from app.api.v1.admin import router as admin_router
from app.api.v1.finance import router as finance_router
from app.api.v1.weather import router as weather_router
from app.api.v1.fortune import router as fortune_router
from app.api.v1.consumption import router as consumption_router
from app.api.v1.ai_crawler import router as ai_crawler_router
from app.api.v1.crawler_task import router as crawler_task_router

from fastapi import APIRouter

router = APIRouter()

router.include_router(auth_router, prefix="/auth", tags=["认证"])
router.include_router(admin_router, prefix="/admin", tags=["系统管理"])
router.include_router(finance_router, prefix="/finance", tags=["金融分析"])
router.include_router(weather_router, prefix="/weather", tags=["气象分析"])
router.include_router(fortune_router, prefix="/fortune", tags=["看相算命"])
router.include_router(consumption_router, prefix="/consumption", tags=["消费分析"])
router.include_router(ai_crawler_router, prefix="/ai-crawler", tags=["AI 智能爬虫"])
router.include_router(crawler_task_router, prefix="/crawler-task", tags=["爬虫任务管理"])
