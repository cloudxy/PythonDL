"""爬虫监控和日志系统

此模块提供爬虫运行状态监控、性能指标记录和异常告警功能。
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import Column, String, Integer, DateTime, Float, Text, Boolean, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import Base
from app.core.logger import get_logger

logger = get_logger("crawler_monitor")


class CrawlerTask(Base):
    """爬虫任务记录表"""
    __tablename__ = "crawler_tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_name = Column(String(100), nullable=False, comment="任务名称")
    crawler_type = Column(String(50), nullable=False, comment="爬虫类型")
    status = Column(String(20), default="pending", comment="状态：pending/running/success/failed")
    start_time = Column(DateTime, comment="开始时间")
    end_time = Column(DateTime, comment="结束时间")
    duration = Column(Float, comment="耗时（秒）")
    total_records = Column(Integer, default=0, comment="采集记录数")
    error_count = Column(Integer, default=0, comment="错误次数")
    error_message = Column(Text, comment="错误信息")
    mode = Column(String(20), default="mock", comment="模式：mock/real_api")
    created_at = Column(DateTime, server_default=func.now())


class CrawlerPerformance(Base):
    """爬虫性能指标表"""
    __tablename__ = "crawler_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crawler_type = Column(String(50), nullable=False, comment="爬虫类型")
    timestamp = Column(DateTime, default=datetime.now, comment="时间戳")
    request_count = Column(Integer, default=0, comment="请求次数")
    success_count = Column(Integer, default=0, comment="成功次数")
    failed_count = Column(Integer, default=0, comment="失败次数")
    cache_hits = Column(Integer, default=0, comment="缓存命中次数")
    cache_misses = Column(Integer, default=0, comment="缓存未命中次数")
    avg_response_time = Column(Float, comment="平均响应时间（秒）")
    records_per_second = Column(Float, comment="每秒记录数")


class CrawlerAlert(Base):
    """爬虫告警记录表"""
    __tablename__ = "crawler_alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False, comment="告警类型")
    severity = Column(String(20), default="warning", comment="严重程度：info/warning/error/critical")
    message = Column(Text, nullable=False, comment="告警信息")
    crawler_type = Column(String(50), comment="爬虫类型")
    is_resolved = Column(Boolean, default=False, comment="是否已解决")
    resolved_at = Column(DateTime, comment="解决时间")
    created_at = Column(DateTime, server_default=func.now())


class CrawlerMonitor:
    """爬虫监控类"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.alert_thresholds = {
            "error_rate": 0.1,  # 错误率超过 10% 告警
            "slow_response": 5.0,  # 响应时间超过 5 秒告警
            "failure_count": 10,  # 连续失败 10 次告警
        }
    
    async def create_task(self, task_name: str, crawler_type: str, mode: str = "mock") -> int:
        """创建爬虫任务记录"""
        task = CrawlerTask(
            task_name=task_name,
            crawler_type=crawler_type,
            status="running",
            start_time=datetime.now(),
            mode=mode
        )
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        logger.info(f"创建爬虫任务：{task_name}, 类型：{crawler_type}")
        return task.id
    
    async def update_task_status(
        self,
        task_id: int,
        status: str,
        total_records: int = 0,
        error_count: int = 0,
        error_message: str = ""
    ):
        """更新爬虫任务状态"""
        stmt = select(CrawlerTask).where(CrawlerTask.id == task_id)
        result = await self.db.execute(stmt)
        task = result.scalar_one_or_none()
        
        if not task:
            logger.error(f"未找到爬虫任务：{task_id}")
            return
        
        task.status = status
        task.total_records = total_records
        task.error_count = error_count
        
        if status in ["success", "failed"]:
            task.end_time = datetime.now()
            if task.start_time:
                task.duration = (task.end_time - task.start_time).total_seconds()
        
        if error_message:
            task.error_message = error_message[:1000]  # 限制长度
        
        await self.db.commit()
        logger.info(f"更新爬虫任务状态：{task_id}, 状态：{status}")
        
        # 检查是否需要告警
        if status == "failed" or error_count > self.alert_thresholds["failure_count"]:
            await self.create_alert(
                alert_type="task_failure",
                severity="error",
                message=f"爬虫任务失败：{task.task_name}, 错误：{error_message}",
                crawler_type=task.crawler_type
            )
    
    async def record_performance(self, crawler_type: str, metrics: Dict[str, Any]):
        """记录性能指标"""
        performance = CrawlerPerformance(
            crawler_type=crawler_type,
            request_count=metrics.get("total_requests", 0),
            success_count=metrics.get("successful_requests", 0),
            failed_count=metrics.get("failed_requests", 0),
            cache_hits=metrics.get("cache_hits", 0),
            cache_misses=metrics.get("cache_misses", 0),
            avg_response_time=metrics.get("avg_response_time", 0),
            records_per_second=metrics.get("records_per_second", 0)
        )
        self.db.add(performance)
        await self.db.commit()
        
        # 检查性能告警
        error_rate = metrics.get("failed_requests", 0) / max(1, metrics.get("total_requests", 1))
        if error_rate > self.alert_thresholds["error_rate"]:
            await self.create_alert(
                alert_type="high_error_rate",
                severity="warning",
                message=f"{crawler_type} 错误率过高：{error_rate:.2%}",
                crawler_type=crawler_type
            )
        
        if metrics.get("avg_response_time", 0) > self.alert_thresholds["slow_response"]:
            await self.create_alert(
                alert_type="slow_response",
                severity="warning",
                message=f"{crawler_type} 响应时间过长：{metrics.get('avg_response_time', 0):.2f}秒",
                crawler_type=crawler_type
            )
    
    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        crawler_type: Optional[str] = None
    ):
        """创建告警记录"""
        alert = CrawlerAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            crawler_type=crawler_type
        )
        self.db.add(alert)
        await self.db.commit()
        logger.warning(f"爬虫告警 [{severity}]: {message}")
    
    async def get_task_history(
        self,
        crawler_type: Optional[str] = None,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """获取任务历史记录"""
        stmt = select(CrawlerTask)
        
        if crawler_type:
            stmt = stmt.where(CrawlerTask.crawler_type == crawler_type)
        
        start_date = datetime.now() - timedelta(days=days)
        stmt = stmt.where(CrawlerTask.created_at >= start_date)
        stmt = stmt.order_by(CrawlerTask.created_at.desc())
        
        result = await self.db.execute(stmt)
        tasks = result.scalars().all()
        
        return [
            {
                "id": task.id,
                "task_name": task.task_name,
                "crawler_type": task.crawler_type,
                "status": task.status,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "end_time": task.end_time.isoformat() if task.end_time else None,
                "duration": task.duration,
                "total_records": task.total_records,
                "error_count": task.error_count,
                "mode": task.mode
            }
            for task in tasks
        ]
    
    async def get_performance_stats(
        self,
        crawler_type: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """获取性能统计"""
        stmt = select(CrawlerPerformance)
        
        if crawler_type:
            stmt = stmt.where(CrawlerPerformance.crawler_type == crawler_type)
        
        start_time = datetime.now() - timedelta(hours=hours)
        stmt = stmt.where(CrawlerPerformance.timestamp >= start_time)
        
        result = await self.db.execute(stmt)
        performances = result.scalars().all()
        
        if not performances:
            return {}
        
        total_requests = sum(p.request_count for p in performances)
        total_success = sum(p.success_count for p in performances)
        total_failed = sum(p.failed_count for p in performances)
        total_cache_hits = sum(p.cache_hits for p in performances)
        total_cache_misses = sum(p.cache_misses for p in performances)
        
        return {
            "total_requests": total_requests,
            "successful_requests": total_success,
            "failed_requests": total_failed,
            "success_rate": total_success / max(1, total_requests),
            "cache_hit_rate": total_cache_hits / max(1, total_cache_hits + total_cache_misses),
            "avg_response_time": sum(p.avg_response_time for p in performances) / len(performances),
            "total_records": sum(p.records_per_second * 3600 for p in performances),
            "sample_count": len(performances)
        }
    
    async def get_active_alerts(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        stmt = select(CrawlerAlert).where(
            CrawlerAlert.is_resolved == False,
            CrawlerAlert.created_at >= datetime.now() - timedelta(days=days)
        )
        stmt = stmt.order_by(CrawlerAlert.created_at.desc())
        
        result = await self.db.execute(stmt)
        alerts = result.scalars().all()
        
        return [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "crawler_type": alert.crawler_type,
                "created_at": alert.created_at.isoformat()
            }
            for alert in alerts
        ]
    
    async def resolve_alert(self, alert_id: int):
        """解决告警"""
        stmt = select(CrawlerAlert).where(CrawlerAlert.id == alert_id)
        result = await self.db.execute(stmt)
        alert = result.scalar_one_or_none()
        
        if alert:
            alert.is_resolved = True
            alert.resolved_at = datetime.now()
            await self.db.commit()
            logger.info(f"已解决告警：{alert_id}")


# 全局监控实例
monitor = None


def get_monitor(db: AsyncSession) -> CrawlerMonitor:
    """获取监控实例"""
    global monitor
    if monitor is None:
        monitor = CrawlerMonitor(db)
    return monitor
