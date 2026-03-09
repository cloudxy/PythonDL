"""爬虫采集API路由

此模块定义爬虫采集相关的API接口。
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.services.crawler.stock_crawler import StockCrawler
from app.services.crawler.weather_crawler import WeatherCrawler
from app.services.crawler.fortune_crawler import FortuneCrawler
from app.services.crawler.consumption_crawler import ConsumptionCrawler

router = APIRouter()


@router.post("/stock/start")
async def start_stock_crawler(
    background_tasks: BackgroundTasks,
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """启动股票数据采集"""
    crawler = StockCrawler(db)
    background_tasks.add_task(crawler.crawl_stock_data, days=days)
    return {"message": "股票数据采集任务已启动", "days": days}


@router.get("/stock/status")
async def get_stock_crawler_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取股票数据采集状态"""
    crawler = StockCrawler(db)
    status = crawler.get_status()
    return status


@router.post("/weather/start")
async def start_weather_crawler(
    background_tasks: BackgroundTasks,
    days: int = 365,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """启动气象数据采集"""
    crawler = WeatherCrawler(db)
    background_tasks.add_task(crawler.crawl_weather_data, days=days)
    return {"message": "气象数据采集任务已启动", "days": days}


@router.get("/weather/status")
async def get_weather_crawler_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取气象数据采集状态"""
    crawler = WeatherCrawler(db)
    status = crawler.get_status()
    return status


@router.post("/fortune/start")
async def start_fortune_crawler(
    background_tasks: BackgroundTasks,
    data_types: list[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """启动看相算命数据采集"""
    crawler = FortuneCrawler(db)
    background_tasks.add_task(crawler.crawl_fortune_data, data_types=data_types)
    return {"message": "看相算命数据采集任务已启动", "data_types": data_types}


@router.get("/fortune/status")
async def get_fortune_crawler_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取看相算命数据采集状态"""
    crawler = FortuneCrawler(db)
    status = crawler.get_status()
    return status


@router.post("/consumption/start")
async def start_consumption_crawler(
    background_tasks: BackgroundTasks,
    data_types: list[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """启动宏观消费数据采集"""
    crawler = ConsumptionCrawler(db)
    background_tasks.add_task(crawler.crawl_consumption_data, data_types=data_types)
    return {"message": "宏观消费数据采集任务已启动", "data_types": data_types}


@router.get("/consumption/status")
async def get_consumption_crawler_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取宏观消费数据采集状态"""
    crawler = ConsumptionCrawler(db)
    status = crawler.get_status()
    return status
