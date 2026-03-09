"""API路由初始化

此模块定义API路由结构。
"""
from fastapi import APIRouter
from app.api.v1 import auth, admin, finance, weather, fortune, consumption, crawler

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(auth.router, prefix="/auth", tags=["认证"])
v1_router.include_router(admin.router, prefix="/admin", tags=["系统管理"])
v1_router.include_router(finance.router, prefix="/finance", tags=["金融分析"])
v1_router.include_router(weather.router, prefix="/weather", tags=["气象分析"])
v1_router.include_router(fortune.router, prefix="/fortune", tags=["看相算命"])
v1_router.include_router(consumption.router, prefix="/consumption", tags=["消费分析"])
v1_router.include_router(crawler.router, prefix="/crawler", tags=["爬虫采集"])
