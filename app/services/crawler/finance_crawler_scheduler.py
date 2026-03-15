"""金融爬虫调度器

此模块提供统一的金融爬虫调度管理，支持：
- 定时任务调度
- 任务优先级管理
- 并发控制
- 任务监控和重试
- 数据源自动切换

基于 ScrapeGraphAI 的图调度架构设计。
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from app.core.logger import get_logger
from app.services.crawler.akshare_crawler import AKShareCrawler, StockData, FinancialIndicator
from app.services.crawler.smart_finance_crawler import SmartFinanceCrawler, FinanceDataExtractionResult
from app.services.crawler.ai_crawler import LLMConfig, CrawlerGraphConfig, CrawlerExecutionState

logger = get_logger("finance_scheduler")


class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class CrawlerTask:
    """爬虫任务定义"""
    id: str
    name: str
    task_type: str  # stock_basic/realtime_quote/history/financial
    symbols: List[str]  # 股票代码列表
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 调度相关
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 执行统计
    retry_count: int = 0
    max_retries: int = 3
    progress: int = 0
    items_collected: int = 0
    error_message: Optional[str] = None
    
    # 回调
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "symbols": self.symbols,
            "priority": self.priority.value,
            "status": self.status.value,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "progress": self.progress,
            "items_collected": self.items_collected,
            "error_message": self.error_message,
        }


@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_concurrent_tasks: int = 5  # 最大并发任务数
    max_concurrent_per_symbol: int = 3  # 单只股票最大并发采集数
    rate_limit_per_second: float = 1.0  # 每秒请求数限制
    retry_delay_seconds: int = 5  # 重试延迟
    task_timeout_seconds: int = 300  # 任务超时时间
    enable_cache: bool = True  # 启用缓存
    cache_ttl_seconds: int = 3600  # 缓存过期时间
    auto_retry: bool = True  # 自动重试
    priority_boost_enabled: bool = True  # 优先级提升
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_concurrent_per_symbol": self.max_concurrent_per_symbol,
            "rate_limit_per_second": self.rate_limit_per_second,
            "retry_delay_seconds": self.retry_delay_seconds,
            "task_timeout_seconds": self.task_timeout_seconds,
            "enable_cache": self.enable_cache,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "auto_retry": self.auto_retry,
            "priority_boost_enabled": self.priority_boost_enabled,
        }


class FinanceCrawlerScheduler:
    """金融爬虫调度器
    
    负责任务调度、并发控制、数据源管理
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        db_session=None,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        初始化调度器
        
        Args:
            config: 调度器配置
            db_session: 数据库会话
            llm_config: LLM 配置
        """
        self.config = config or SchedulerConfig()
        self.db_session = db_session
        self.llm_config = llm_config
        
        # 任务队列
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, CrawlerTask] = {}
        self.completed_tasks: Dict[str, CrawlerTask] = {}
        self.failed_tasks: Dict[str, CrawlerTask] = {}
        
        # 信号量控制
        self.task_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self.symbol_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # 数据缓存
        self.data_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # 运行状态
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        logger.info("金融爬虫调度器初始化完成")
    
    async def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("调度器已在运行中")
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        if not self._running:
            return
        
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有运行中的任务
        for task_id, task in list(self.running_tasks.items()):
            await self._cancel_task(task_id)
        
        logger.info("调度器已停止")
    
    async def submit_task(self, task: CrawlerTask) -> str:
        """
        提交任务
        
        Args:
            task: 爬虫任务
            
        Returns:
            任务 ID
        """
        # 设置调度时间
        if not task.scheduled_at:
            task.scheduled_at = datetime.now()
        
        # 添加到优先队列
        priority_value = self._get_priority_value(task.priority)
        await self.pending_queue.put((priority_value, task))
        
        self.stats["total_tasks"] += 1
        logger.info(f"任务已提交：{task.id} - {task.name}")
        
        return task.id
    
    async def submit_batch(
        self,
        task_type: str,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> List[str]:
        """
        批量提交任务
        
        Args:
            task_type: 任务类型
            symbols: 股票代码列表
            config: 任务配置
            priority: 优先级
            
        Returns:
            任务 ID 列表
        """
        task_ids = []
        
        # 按优先级和类型分组
        batch_task = CrawlerTask(
            id=f"{task_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"批量采集 - {task_type}",
            task_type=task_type,
            symbols=symbols,
            priority=priority,
            config=config or {}
        )
        
        task_id = await self.submit_task(batch_task)
        task_ids.append(task_id)
        
        logger.info(f"批量任务已提交：{len(symbols)}只股票")
        
        return task_ids
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            是否成功取消
        """
        # 从队列中移除
        # 注意：PriorityQueue 不支持直接移除，需要标记为取消
        
        # 从运行中取消
        if task_id in self.running_tasks:
            await self._cancel_task(task_id)
            return True
        
        # 从已完成中移除
        if task_id in self.completed_tasks:
            logger.warning(f"任务已完成，无法取消：{task_id}")
            return False
        
        if task_id in self.failed_tasks:
            logger.warning(f"任务已失败，无法取消：{task_id}")
            return False
        
        logger.warning(f"未找到任务：{task_id}")
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务状态字典
        """
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].to_dict()
        
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id].to_dict()
        
        return None
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据
        """
        if not self.config.enable_cache:
            return None
        
        if key in self.data_cache:
            # 检查是否过期
            if key in self.cache_timestamps:
                age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
                if age > self.config.cache_ttl_seconds:
                    # 缓存过期
                    del self.data_cache[key]
                    del self.cache_timestamps[key]
                    self.stats["cache_misses"] += 1
                    return None
            
            self.stats["cache_hits"] += 1
            return self.data_cache[key]
        
        self.stats["cache_misses"] += 1
        return None
    
    async def set_cache(self, key: str, value: Any):
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if not self.config.enable_cache:
            return
        
        self.data_cache[key] = value
        self.cache_timestamps[key] = datetime.now()
        logger.debug(f"缓存已设置：{key}")
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        清空缓存
        
        Args:
            pattern: 匹配模式，None 表示清空所有
        """
        if pattern:
            keys_to_remove = [k for k in self.data_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.data_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
            logger.info(f"清空缓存：{len(keys_to_remove)}条记录")
        else:
            count = len(self.data_cache)
            self.data_cache.clear()
            self.cache_timestamps.clear()
            logger.info(f"清空所有缓存：{count}条记录")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "pending_tasks": self.pending_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "cache_size": len(self.data_cache),
        }
    
    async def _run_scheduler(self):
        """调度器主循环"""
        while self._running:
            try:
                # 获取任务
                try:
                    priority, task = await asyncio.wait_for(
                        self.pending_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 检查是否已取消
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # 检查调度时间
                if task.scheduled_at and task.scheduled_at > datetime.now():
                    # 还未到调度时间，重新加入队列
                    await self.pending_queue.put((priority, task))
                    await asyncio.sleep(0.1)
                    continue
                
                # 执行任务
                asyncio.create_task(self._execute_task(task))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度器错误：{e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: CrawlerTask):
        """
        执行任务
        
        Args:
            task: 爬虫任务
        """
        # 获取信号量
        await self.task_semaphore.acquire()
        
        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks[task.id] = task
            
            logger.info(f"开始执行任务：{task.id} - {task.name}")
            
            # 创建爬虫
            crawler = SmartFinanceCrawler(
                llm_config=self.llm_config,
                graph_config=CrawlerGraphConfig(),
                db_session=self.db_session
            )
            
            # 准备配置
            config = {
                "id": task.id,
                "name": task.name,
                "data_type": task.task_type,
                "symbols": task.symbols,
                "sources": task.config.get("sources", [{"type": "akshare"}]),
                **task.config
            }
            
            # 执行爬虫（带超时）
            try:
                result = await asyncio.wait_for(
                    crawler.run(config),
                    timeout=self.config.task_timeout_seconds
                )
                
                # 更新任务状态
                task.progress = result.progress
                task.items_collected = result.items_collected
                
                if result.status == "completed":
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    self.completed_tasks[task.id] = task
                    self.stats["completed_tasks"] += 1
                    logger.info(f"任务完成：{task.id} - 采集{result.items_collected}条数据")
                else:
                    raise Exception(result.error_message or "爬虫执行失败")
                
            except asyncio.TimeoutError:
                raise Exception(f"任务超时（{self.config.task_timeout_seconds}秒）")
            
            # 调用完成回调
            if task.on_complete:
                await task.on_complete(task)
            
        except Exception as e:
            logger.error(f"任务执行失败：{task.id} - {e}")
            task.error_message = str(e)
            
            # 重试逻辑
            if self.config.auto_retry and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.stats["retried_tasks"] += 1
                
                # 延迟重试
                retry_delay = self.config.retry_delay_seconds * (2 ** task.retry_count)
                task.scheduled_at = datetime.now() + timedelta(seconds=retry_delay)
                await self.pending_queue.put((self._get_priority_value(task.priority), task))
                
                logger.info(f"任务将重试：{task.id} (第{task.retry_count}次，{retry_delay}秒后)")
                
                if task.on_error:
                    await task.on_error(task, e, True)
            else:
                # 失败
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                self.failed_tasks[task.id] = task
                self.stats["failed_tasks"] += 1
                
                if task.on_error:
                    await task.on_error(task, e, False)
        
        finally:
            # 释放信号量
            self.task_semaphore.release()
            
            # 从运行中移除
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _cancel_task(self, task_id: str):
        """取消任务"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            logger.info(f"任务已取消：{task_id}")
    
    def _get_priority_value(self, priority: TaskPriority) -> int:
        """获取优先级数值（数值越小优先级越高）"""
        priority_map = {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
        }
        return priority_map.get(priority, 2)
    
    def _get_symbol_semaphore(self, symbol: str) -> asyncio.Semaphore:
        """获取股票代码的信号量"""
        if symbol not in self.symbol_semaphores:
            self.symbol_semaphores[symbol] = asyncio.Semaphore(
                self.config.max_concurrent_per_symbol
            )
        return self.symbol_semaphores[symbol]


# 便捷函数
async def create_scheduler(
    db_session=None,
    llm_config: Optional[LLMConfig] = None,
    **kwargs
) -> FinanceCrawlerScheduler:
    """
    创建调度器的便捷函数
    
    Args:
        db_session: 数据库会话
        llm_config: LLM 配置
        **kwargs: 其他配置参数
        
    Returns:
        FinanceCrawlerScheduler 实例
    """
    config = SchedulerConfig(**kwargs) if kwargs else None
    scheduler = FinanceCrawlerScheduler(
        config=config,
        db_session=db_session,
        llm_config=llm_config
    )
    return scheduler
