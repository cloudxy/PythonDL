"""爬虫任务调度器

此模块提供爬虫任务的定时调度和自动执行功能。
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
import json

from app.core.logger import get_logger
from app.models.admin.crawler_task import CrawlerTask, CrawlerLog, TaskStatus
from app.services.crawler.real_stock_crawler import real_stock_crawler
from app.services.crawler.crawler_monitor import CrawlerMonitor, get_monitor

logger = get_logger("crawler_scheduler")


class CrawlerScheduler:
    """爬虫任务调度器"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.monitor = get_monitor(db)
        self.running_tasks: Dict[int, asyncio.Task] = {}
        self.is_running = False
    
    async def start(self):
        """启动调度器"""
        logger.info("爬虫任务调度器启动")
        self.is_running = True
        
        # 启动调度循环
        asyncio.create_task(self._schedule_loop())
    
    async def stop(self):
        """停止调度器"""
        logger.info("停止爬虫任务调度器")
        self.is_running = False
        
        # 停止所有运行中的任务
        for task_id, task in self.running_tasks.items():
            task.cancel()
            logger.info(f"已取消任务：{task_id}")
        
        self.running_tasks.clear()
    
    async def _schedule_loop(self):
        """调度循环"""
        while self.is_running:
            try:
                # 每分钟检查一次需要执行的任务
                await self._check_and_run_tasks()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度循环错误：{e}")
                await asyncio.sleep(60)
    
    async def _check_and_run_tasks(self):
        """检查并执行到期的任务"""
        try:
            # 查询所有激活的定时任务
            stmt = select(CrawlerTask).where(
                CrawlerTask.is_active == True,
                CrawlerTask.schedule_type != "manual",
                CrawlerTask.status != TaskStatus.RUNNING
            )
            
            result = await self.db.execute(stmt)
            tasks = result.scalars().all()
            
            now = datetime.now()
            
            for task in tasks:
                # 检查是否需要执行
                should_run = False
                
                if task.schedule_type == "interval":
                    # 间隔调度
                    if task.last_run_at:
                        next_run = task.last_run_at + timedelta(seconds=task.interval_seconds)
                        if now >= next_run:
                            should_run = True
                    else:
                        # 从未执行过，立即执行
                        should_run = True
                
                elif task.schedule_type == "cron":
                    # Cron 调度（简化实现）
                    if self._match_cron(task.cron_expression, now):
                        should_run = True
                
                if should_run:
                    # 执行任务
                    asyncio.create_task(self._execute_task(task.id))
        
        except Exception as e:
            logger.error(f"检查任务错误：{e}")
    
    def _match_cron(self, cron_expr: str, dt: datetime) -> bool:
        """匹配 Cron 表达式（简化版）"""
        if not cron_expr:
            return False
        
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                return False
            
            minute, hour, day, month, weekday = parts
            
            # 简化匹配逻辑
            if minute != "*" and int(minute) != dt.minute:
                return False
            if hour != "*" and int(hour) != dt.hour:
                return False
            if day != "*" and int(day) != dt.day:
                return False
            if month != "*" and int(month) != dt.month:
                return False
            if weekday != "*" and int(weekday) != dt.weekday():
                return False
            
            return True
        except Exception:
            return False
    
    async def _execute_task(self, task_id: int):
        """执行爬虫任务"""
        try:
            # 获取任务信息
            stmt = select(CrawlerTask).where(CrawlerTask.id == task_id)
            result = await self.db.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                logger.error(f"任务不存在：{task_id}")
                return
            
            # 检查任务是否正在运行
            if task.is_running:
                logger.warning(f"任务正在运行：{task_id}")
                return
            
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.is_running = True
            await self.db.commit()
            
            logger.info(f"开始执行任务：{task_id}, 类型：{task.task_type}")
            
            # 创建监控任务
            monitor_task_id = await self.monitor.create_task(
                task_name=task.name,
                crawler_type=task.task_type,
                mode="auto"
            )
            
            # 执行对应的爬虫
            start_time = datetime.now()
            success = await self._run_crawler(task)
            end_time = datetime.now()
            
            # 更新任务状态
            if success:
                task.status = TaskStatus.SUCCESS
                task.success_runs += 1
            else:
                task.status = TaskStatus.FAILED
                task.failed_runs += 1
            
            task.last_run_at = start_time
            task.total_runs += 1
            task.is_running = False
            
            # 计算下次执行时间
            if task.schedule_type == "interval":
                task.next_run_at = start_time + timedelta(seconds=task.interval_seconds)
            
            await self.db.commit()
            
            # 更新监控任务状态
            await self.monitor.update_task_status(
                task_id=monitor_task_id,
                status="success" if success else "failed",
                total_records=task.total_records
            )
            
            # 记录日志
            await self._create_log(
                task_id=task.id,
                task_name=task.name,
                log_level="INFO" if success else "ERROR",
                message=f"任务执行{'成功' if success else '失败'}",
                execution_id=str(monitor_task_id),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds()
            )
            
            logger.info(f"任务执行完成：{task_id}, 结果：{'成功' if success else '失败'}")
            
        except Exception as e:
            logger.error(f"执行任务失败：{task_id}, 错误：{e}")
            
            # 更新任务状态
            try:
                stmt = update(CrawlerTask).where(
                    CrawlerTask.id == task_id
                ).values(
                    status=TaskStatus.FAILED,
                    is_running=False,
                    failed_runs=CrawlerTask.failed_runs + 1
                )
                await self.db.execute(stmt)
                await self.db.commit()
            except Exception as update_error:
                logger.error(f"更新任务状态失败：{update_error}")
    
    async def _run_crawler(self, task: CrawlerTask) -> bool:
        """运行爬虫"""
        try:
            task_type = task.task_type
            config = task.crawler_config or {}
            
            if task_type == "stock":
                # 执行股票爬虫
                stocks = real_stock_crawler.get_stock_basics_from_sina()
                task.total_records = len(stocks)
                return len(stocks) > 0
            
            elif task_type == "weather":
                # 执行气象爬虫（需要数据库 session）
                # 这里简化处理
                task.total_records = 0
                return True
            
            elif task_type == "consumption":
                # 执行消费数据爬虫
                task.total_records = 0
                return True
            
            else:
                logger.warning(f"未知任务类型：{task_type}")
                return False
        
        except Exception as e:
            logger.error(f"爬虫执行失败：{e}")
            return False
    
    async def _create_log(
        self,
        task_id: int,
        task_name: str,
        log_level: str,
        message: str,
        execution_id: str,
        start_time: datetime,
        end_time: datetime,
        duration_seconds: float
    ):
        """创建日志记录"""
        try:
            log = CrawlerLog(
                task_id=task_id,
                task_name=task_name,
                log_level=log_level,
                message=message,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds
            )
            self.db.add(log)
            await self.db.commit()
        except Exception as e:
            logger.error(f"创建日志失败：{e}")
    
    async def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """获取所有定时任务"""
        try:
            stmt = select(CrawlerTask).where(
                CrawlerTask.schedule_type != "manual"
            ).order_by(CrawlerTask.created_at.desc())
            
            result = await self.db.execute(stmt)
            tasks = result.scalars().all()
            
            return [
                {
                    "id": task.id,
                    "name": task.name,
                    "task_type": task.task_type,
                    "schedule_type": task.schedule_type,
                    "cron_expression": task.cron_expression,
                    "interval_seconds": task.interval_seconds,
                    "is_active": task.is_active,
                    "last_run_at": task.last_run_at.isoformat() if task.last_run_at else None,
                    "next_run_at": task.next_run_at.isoformat() if task.next_run_at else None,
                    "total_runs": task.total_runs,
                    "success_runs": task.success_runs,
                    "failed_runs": task.failed_runs
                }
                for task in tasks
            ]
        except Exception as e:
            logger.error(f"获取定时任务失败：{e}")
            return []


# 全局调度器实例
_scheduler: Optional[CrawlerScheduler] = None


def get_scheduler(db: AsyncSession) -> CrawlerScheduler:
    """获取调度器实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = CrawlerScheduler(db)
    return _scheduler
