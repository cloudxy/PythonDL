"""爬虫任务管理服务

此模块提供爬虫任务管理业务逻辑。
"""
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import joinedload
from datetime import datetime, timedelta
import uuid
import logging

from app.models.admin.crawler_task import CrawlerTask, CrawlerLog, TaskStatus, CrawlerData
from app.schemas.admin.crawler_task import CrawlerTaskCreate, CrawlerTaskUpdate

logger = logging.getLogger(__name__)


class CrawlerTaskService:
    """爬虫任务服务类"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_tasks(
        self,
        page: int = 1,
        page_size: int = 20,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Tuple[List[Dict], int]:
        """获取任务列表"""
        # 构建查询
        query = select(CrawlerTask)
        
        # 添加筛选条件
        if task_type:
            query = query.where(CrawlerTask.task_type == task_type)
        if status:
            query = query.where(CrawlerTask.status == status)
        if is_active is not None:
            query = query.where(CrawlerTask.is_active == is_active)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # 分页查询
        offset = (page - 1) * page_size
        query = query.order_by(CrawlerTask.created_at.desc()).offset(offset).limit(page_size)
        
        result = await self.db.execute(query)
        tasks = result.scalars().all()
        
        # 转换为字典
        task_list = [self._task_to_dict(task) for task in tasks]
        
        return task_list, total
    
    async def get_task(self, task_id: int) -> Optional[Dict]:
        """获取任务详情"""
        query = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(query)
        task = result.scalar_one_or_none()
        
        if task:
            return self._task_to_dict(task)
        return None
    
    async def create_task(self, task_data: CrawlerTaskCreate) -> Dict:
        """创建任务"""
        task = CrawlerTask(**task_data.model_dump())
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"创建爬虫任务：{task.name}, ID: {task.id}")
        return self._task_to_dict(task)
    
    async def update_task(self, task_id: int, task_data: CrawlerTaskUpdate) -> Optional[Dict]:
        """更新任务"""
        query = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            return None
        
        # 更新字段
        update_data = task_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)
        
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"更新爬虫任务：{task.name}, ID: {task.id}")
        return self._task_to_dict(task)
    
    async def delete_task(self, task_id: int) -> bool:
        """删除任务"""
        query = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            return False
        
        await self.db.delete(task)
        await self.db.commit()
        
        logger.info(f"删除爬虫任务：{task.name}, ID: {task.id}")
        return True
    
    async def start_task(self, task_id: int) -> Optional[Dict]:
        """启动任务"""
        query = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            return None
        
        task.is_active = True
        task.status = TaskStatus.PENDING
        
        # 计算下次执行时间
        if task.schedule_type == "interval":
            task.next_run_at = datetime.now() + timedelta(seconds=task.interval_seconds)
        elif task.schedule_type == "cron":
            # TODO: 解析 cron 表达式
            task.next_run_at = datetime.now() + timedelta(hours=1)
        
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"启动爬虫任务：{task.name}, ID: {task.id}")
        return self._task_to_dict(task)
    
    async def stop_task(self, task_id: int) -> Optional[Dict]:
        """停止任务"""
        query = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            return None
        
        task.is_active = False
        task.is_running = False
        task.status = TaskStatus.STOPPED
        
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"停止爬虫任务：{task.name}, ID: {task.id}")
        return self._task_to_dict(task)
    
    async def run_task_now(self, task_id: int) -> str:
        """立即执行任务"""
        query = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise ValueError(f"任务不存在：{task_id}")
        
        # 生成执行 ID
        execution_id = f"exec_{task_id}_{uuid.uuid4().hex[:8]}"
        
        # 更新任务状态
        task.status = TaskStatus.RUNNING
        task.is_running = True
        task.last_run_at = datetime.now()
        task.total_runs += 1
        
        # 创建执行日志
        log = CrawlerLog(
            task_id=task_id,
            task_name=task.name,
            log_level="INFO",
            message="任务开始执行",
            execution_id=execution_id,
            start_time=datetime.now()
        )
        self.db.add(log)
        
        await self.db.commit()
        
        logger.info(f"爬虫任务立即执行：{task.name}, ID: {task.id}, Execution ID: {execution_id}")
        
        # TODO: 异步执行爬虫逻辑
        
        return execution_id
    
    async def list_logs(
        self,
        task_id: int,
        page: int = 1,
        page_size: int = 20,
        log_level: Optional[str] = None
    ) -> Tuple[List[Dict], int]:
        """获取日志列表"""
        # 构建查询
        query = select(CrawlerLog).where(CrawlerLog.task_id == task_id)
        
        if log_level:
            query = query.where(CrawlerLog.log_level == log_level)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # 分页查询
        offset = (page - 1) * page_size
        query = query.order_by(CrawlerLog.created_at.desc()).offset(offset).limit(page_size)
        
        result = await self.db.execute(query)
        logs = result.scalars().all()
        
        log_list = [self._log_to_dict(log) for log in logs]
        
        return log_list, total
    
    async def get_log(self, log_id: int) -> Optional[Dict]:
        """获取日志详情"""
        query = select(CrawlerLog).where(CrawlerLog.id == log_id)
        result = await self.db.execute(query)
        log = result.scalar_one_or_none()
        
        if log:
            return self._log_to_dict(log)
        return None
    
    async def get_task_status(self, task_id: int) -> Dict:
        """获取任务状态"""
        task = await self.get_task(task_id)
        
        if not task:
            raise ValueError(f"任务不存在：{task_id}")
        
        # 获取最近执行日志
        query = select(CrawlerLog).where(
            CrawlerLog.task_id == task_id
        ).order_by(CrawlerLog.created_at.desc()).limit(1)
        
        result = await self.db.execute(query)
        last_log = result.scalar_one_or_none()
        
        return {
            "task_id": task_id,
            "status": task["status"],
            "is_active": task["is_active"],
            "is_running": task["is_running"],
            "last_run_at": task.get("last_run_at"),
            "next_run_at": task.get("next_run_at"),
            "total_runs": task.get("total_runs", 0),
            "success_runs": task.get("success_runs", 0),
            "failed_runs": task.get("failed_runs", 0),
            "last_log": self._log_to_dict(last_log) if last_log else None
        }
    
    def _task_to_dict(self, task: CrawlerTask) -> Dict:
        """任务对象转字典"""
        return {
            "id": task.id,
            "name": task.name,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status.value if task.status else None,
            "schedule_type": task.schedule_type,
            "cron_expression": task.cron_expression,
            "interval_seconds": task.interval_seconds,
            "last_run_at": task.last_run_at,
            "next_run_at": task.next_run_at,
            "total_runs": task.total_runs,
            "success_runs": task.success_runs,
            "failed_runs": task.failed_runs,
            "total_records": task.total_records,
            "is_active": task.is_active,
            "is_running": task.is_running,
            "created_at": task.created_at,
            "updated_at": task.updated_at
        }
    
    def _log_to_dict(self, log: CrawlerLog) -> Dict:
        """日志对象转字典"""
        return {
            "id": log.id,
            "task_id": log.task_id,
            "task_name": log.task_name,
            "log_level": log.log_level,
            "message": log.message,
            "details": log.details,
            "execution_id": log.execution_id,
            "start_time": log.start_time,
            "end_time": log.end_time,
            "duration_seconds": log.duration_seconds,
            "records_fetched": log.records_fetched,
            "records_saved": log.records_saved,
            "error_count": log.error_count,
            "error_message": log.error_message,
            "created_at": log.created_at
        }
