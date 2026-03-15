"""AI 爬虫 API 接口

此模块提供 AI 爬虫的 API 接口，包括爬虫执行、状态监控等。
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.services.crawler.ai_crawler import (
    SmartScraperCrawler,
    SearchGraphCrawler,
    CrawlerGraphConfig,
    LLMConfig,
    LLMProvider,
)
from app.services.crawler.crawler_config import (
    get_config_manager,
    CrawlerType,
)
from app.core.logger import get_logger

logger = get_logger("ai_crawler_api")

router = APIRouter()

# 存储运行中的爬虫实例
_running_crawlers: Dict[str, Dict[str, Any]] = {}


@router.post("/ai/smart-scraper")
async def create_smart_scraper_task(
    prompt: str,
    source: str,
    llm_provider: str = "ollama",
    llm_model: str = "llama3.2",
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """创建智能单页爬虫任务
    
    - **prompt**: 自然语言描述需要提取的信息
    - **source**: 目标网页 URL
    - **llm_provider**: LLM 提供商 (ollama, openai, groq)
    - **llm_model**: LLM 模型名称
    """
    try:
        # 创建配置
        llm_config = LLMConfig(
            provider=LLMProvider(llm_provider),
            model=llm_model,
            format="json"
        )
        
        config = CrawlerGraphConfig(
            llm=llm_config,
            verbose=True,
            max_concurrent=3
        )
        
        # 创建爬虫实例
        crawler = SmartScraperCrawler(
            prompt=prompt,
            source=source,
            config=config
        )
        
        # 生成任务 ID
        task_id = f"smart_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 存储爬虫状态
        _running_crawlers[task_id] = {
            "crawler": crawler,
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        # 在后台执行爬虫
        async def run_crawler():
            try:
                _running_crawlers[task_id]["status"] = "running"
                await crawler.initialize()
                result = await crawler.crawl()
                _running_crawlers[task_id]["status"] = "completed"
                _running_crawlers[task_id]["result"] = result
                await crawler.close()
            except Exception as e:
                logger.error(f"爬虫执行失败：{e}")
                _running_crawlers[task_id]["status"] = "failed"
                _running_crawlers[task_id]["error"] = str(e)
        
        if background_tasks:
            background_tasks.add_task(run_crawler)
        else:
            # 如果没有 background_tasks，直接执行（用于测试）
            import asyncio
            asyncio.create_task(run_crawler())
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "AI 爬虫任务已创建",
            "config": {
                "prompt": prompt,
                "source": source,
                "llm": f"{llm_provider}/{llm_model}"
            }
        }
        
    except Exception as e:
        logger.error(f"创建 AI 爬虫任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/search-graph")
async def create_search_graph_task(
    prompt: str,
    query: str,
    search_engine: str = "google",
    max_results: int = 10,
    llm_provider: str = "ollama",
    llm_model: str = "llama3.2",
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("crawler:start"))
):
    """创建搜索引擎爬虫任务
    
    - **prompt**: 自然语言描述需要提取的信息
    - **query**: 搜索关键词
    - **search_engine**: 搜索引擎 (google, baidu, bing)
    - **max_results**: 最大搜索结果数
    """
    try:
        # 创建配置
        llm_config = LLMConfig(
            provider=LLMProvider(llm_provider),
            model=llm_model,
            format="json"
        )
        
        config = CrawlerGraphConfig(
            llm=llm_config,
            verbose=True,
            max_results=max_results,
            max_concurrent=3
        )
        
        # 创建爬虫实例
        crawler = SearchGraphCrawler(
            prompt=prompt,
            query=query,
            search_engine=search_engine,
            max_results=max_results,
            config=config
        )
        
        # 生成任务 ID
        task_id = f"search_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 存储爬虫状态
        _running_crawlers[task_id] = {
            "crawler": crawler,
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        # 在后台执行爬虫
        async def run_crawler():
            try:
                _running_crawlers[task_id]["status"] = "running"
                await crawler.initialize()
                result = await crawler.crawl()
                _running_crawlers[task_id]["status"] = "completed"
                _running_crawlers[task_id]["result"] = result
                await crawler.close()
            except Exception as e:
                logger.error(f"爬虫执行失败：{e}")
                _running_crawlers[task_id]["status"] = "failed"
                _running_crawlers[task_id]["error"] = str(e)
        
        if background_tasks:
            background_tasks.add_task(run_crawler)
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "AI 搜索爬虫任务已创建",
            "config": {
                "prompt": prompt,
                "query": query,
                "search_engine": search_engine,
                "max_results": max_results,
                "llm": f"{llm_provider}/{llm_model}"
            }
        }
        
    except Exception as e:
        logger.error(f"创建搜索爬虫任务失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/task/{task_id}/status")
async def get_ai_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """获取 AI 爬虫任务状态"""
    if task_id not in _running_crawlers:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_data = _running_crawlers[task_id]
    crawler = task_data["crawler"]
    
    # 获取爬虫执行状态
    state = crawler.get_state()
    
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "status": task_data["status"],
            "created_at": task_data["created_at"].isoformat(),
            "execution_state": state,
            "result": task_data.get("result"),
            "error": task_data.get("error")
        }
    }


@router.get("/ai/task/{task_id}/result")
async def get_ai_task_result(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """获取 AI 爬虫任务结果"""
    if task_id not in _running_crawlers:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_data = _running_crawlers[task_id]
    
    if task_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态：{task_data['status']}"
        )
    
    return {
        "success": True,
        "data": task_data.get("result", {})
    }


@router.get("/ai/configs")
async def get_ai_crawler_configs(
    current_user: User = Depends(get_current_user)
):
    """获取 AI 爬虫可用配置"""
    try:
        manager = get_config_manager()
        profiles = manager.get_all_profiles()
        
        return {
            "success": True,
            "data": {
                "profiles": profiles,
                "llm_providers": [
                    {"value": "ollama", "label": "Ollama (本地)", "default_model": "llama3.2"},
                    {"value": "openai", "label": "OpenAI", "default_model": "gpt-4o-mini"},
                    {"value": "groq", "label": "Groq", "default_model": "llama3-70b-8192"},
                ]
            }
        }
    except Exception as e:
        logger.error(f"获取配置失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/config/custom")
async def create_custom_crawler_config(
    crawler_type: str,
    name: str,
    description: str,
    data_sources: List[Dict[str, Any]],
    prompt: Dict[str, Any],
    current_user: User = Depends(require_permission("admin:config"))
):
    """创建自定义 AI 爬虫配置"""
    try:
        manager = get_config_manager()
        
        # 转换数据源配置
        from app.services.crawler.crawler_config import DataSourceConfig, ExtractionPrompt
        source_configs = [
            DataSourceConfig(
                name=ds["name"],
                url=ds["url"],
                method=ds.get("method", "GET"),
                headers=ds.get("headers", {}),
                params=ds.get("params", {}),
                timeout=ds.get("timeout", 30),
                cache_ttl=ds.get("cache_ttl", 3600)
            )
            for ds in data_sources
        ]
        
        # 转换 prompt 配置
        prompt_config = ExtractionPrompt(
            name=prompt.get("name", "custom"),
            description=prompt.get("description", ""),
            prompt=prompt.get("prompt", ""),
            output_format=prompt.get("output_format", "json"),
            fields=prompt.get("fields", [])
        )
        
        # 创建自定义配置
        profile = manager.create_custom_profile(
            crawler_type=CrawlerType(crawler_type),
            name=name,
            description=description,
            data_sources=source_configs,
            prompt=prompt_config
        )
        
        return {
            "success": True,
            "message": "自定义配置已创建",
            "profile": profile.to_dict()
        }
    except Exception as e:
        logger.error(f"创建自定义配置失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
