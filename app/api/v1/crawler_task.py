"""爬虫任务管理 API

此模块提供爬虫任务的管理接口，包括启动、停止、查看状态等。
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.services.crawler.crawler_config import get_config_manager, CrawlerType
from app.services.crawler.ai_crawler import (
    SmartScraperCrawler,
    CrawlerGraphConfig,
    LLMConfig,
    LLMProvider,
)
from app.core.logger import get_logger

logger = get_logger("crawler_task_api")

router = APIRouter()

# 存储运行中的任务
_running_tasks: Dict[int, Dict[str, Any]] = {}


@router.post("/start")
async def start_crawler_task(
    task_id: int,
    crawler_type: str,
    config: Dict[str, Any],
    timeout: int = 300,
    retry_times: int = 3,
    cache_ttl: int = 3600,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """启动爬虫任务
    
    - **task_id**: 任务 ID
    - **crawler_type**: 爬虫类型 (stock, weather, consumption, fortune)
    - **config**: 爬虫配置参数
    - **timeout**: 超时时间（秒）
    - **retry_times**: 重试次数
    - **cache_ttl**: 缓存时间（秒）
    """
    try:
        # 获取爬虫配置
        config_manager = get_config_manager()
        profile = config_manager.get_profile(CrawlerType(crawler_type))
        
        if not profile:
            raise HTTPException(status_code=400, detail=f"不支持的爬虫类型：{crawler_type}")
        
        # 创建爬虫配置
        graph_config = CrawlerGraphConfig(
            llm=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=profile.llm_model,
                format="json"
            ),
            verbose=True,
            max_concurrent=profile.rate_limit.concurrent_limit if profile.rate_limit else 5,
            cache_enabled=True,
            dedup_enabled=True
        )
        
        # 根据类型创建不同的爬虫
        if crawler_type == "stock":
            # 股票爬虫
            symbols = config.get("symbols", "000001")
            prompt = "提取股票代码、名称、开盘价、最高价、最低价、收盘价、成交量、成交额"
            
            crawler = SmartScraperCrawler(
                prompt=prompt,
                source="http://hq.sinajs.cn/list=" + symbols.split(",")[0],
                config=graph_config
            )
            
        elif crawler_type == "weather":
            # 气象爬虫
            cities = config.get("cities", "北京")
            prompt = "提取城市名称、温度、湿度、天气状况、风向、风力"
            
            crawler = SmartScraperCrawler(
                prompt=prompt,
                source="http://www.weather.com.cn/weather/101010100.shtml",
                config=graph_config
            )
            
        elif crawler_type == "consumption":
            # 消费数据爬虫
            indicator_type = config.get("indicator_type", "gdp")
            prompt = "提取指标名称、年份、数值、增长率"
            
            crawler = SmartScraperCrawler(
                prompt=prompt,
                source="https://data.stats.gov.cn/easyquery.htm",
                config=graph_config
            )
            
        elif crawler_type == "fortune":
            # 算命数据爬虫
            category = config.get("category", "zhouyi")
            prompt = "提取标题、内容、解释、出处"
            
            crawler = SmartScraperCrawler(
                prompt=prompt,
                source="https://baike.baidu.com/item/周易",
                config=graph_config
            )
            
        else:
            # 通用爬虫
            crawler = SmartScraperCrawler(
                prompt="提取网页主要内容",
                source="https://example.com",
                config=graph_config
            )
        
        # 存储任务信息
        _running_tasks[task_id] = {
            "crawler": crawler,
            "status": "running",
            "progress": 0,
            "started_at": datetime.now(),
            "config": config
        }
        
        # 在后台执行爬虫
        async def run_crawler():
            try:
                task_info = _running_tasks[task_id]
                task_info["status"] = "running"
                task_info["progress"] = 10
                
                await crawler.initialize()
                task_info["progress"] = 30
                
                result = await crawler.crawl()
                task_info["progress"] = 90
                
                # 保存结果到数据库
                # TODO: 实现数据入库逻辑
                
                task_info["progress"] = 100
                task_info["status"] = "completed"
                task_info["result"] = result
                task_info["completed_at"] = datetime.now()
                
                await crawler.close()
                
            except Exception as e:
                logger.error(f"爬虫执行失败：{e}")
                if task_id in _running_tasks:
                    _running_tasks[task_id]["status"] = "failed"
                    _running_tasks[task_id]["error"] = str(e)
        
        if background_tasks:
            background_tasks.add_task(run_crawler)
        else:
            asyncio.create_task(run_crawler())
        
        return {
            "success": True,
            "message": "爬虫任务已启动",
            "task_id": task_id,
            "crawler_type": crawler_type,
            "config": config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动爬虫任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: int,
    current_user: User = Depends(get_current_user)
):
    """获取任务状态"""
    if task_id not in _running_tasks:
        # 返回默认状态
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "unknown",
                "progress": 0
            }
        }
    
    task_info = _running_tasks[task_id]
    
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "status": task_info["status"],
            "progress": task_info["progress"],
            "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
            "completed_at": task_info.get("completed_at").isoformat() if task_info.get("completed_at") else None,
            "error": task_info.get("error"),
            "result": task_info.get("result")
        }
    }


@router.post("/task/{task_id}/stop")
async def stop_task(
    task_id: int,
    current_user: User = Depends(require_permission("crawler:stop"))
):
    """停止任务"""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_info = _running_tasks[task_id]
    task_info["status"] = "stopped"
    
    # TODO: 实现爬虫停止逻辑
    
    return {
        "success": True,
        "message": "任务已停止"
    }


@router.get("/tasks")
async def list_tasks(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user)
):
    """获取任务列表"""
    # TODO: 从数据库获取任务列表
    return {
        "success": True,
        "data": {
            "tasks": [],
            "total": 0,
            "page": page,
            "page_size": page_size
        }
    }
