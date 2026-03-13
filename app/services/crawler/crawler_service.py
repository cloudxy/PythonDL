from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.crawler import CrawlerTask, CrawlerLog, CrawlerData
from datetime import datetime


class CrawlerTaskService:
    def __init__(self, db: Session):
        self.db = db

    def create_task(self, task_data: dict) -> CrawlerTask:
        try:
            task = CrawlerTask(
                name=task_data.get("name"),
                url=task_data.get("url"),
                crawler_type=task_data.get("crawler_type"),
                method=task_data.get("method", "GET"),
                headers=task_data.get("headers"),
                params=task_data.get("params"),
                parse_rules=task_data.get("parse_rules"),
                schedule=task_data.get("schedule"),
                is_active=task_data.get("is_active", True),
                created_at=datetime.now()
            )
            self.db.add(task)
            self.db.commit()
            self.db.refresh(task)
            return task
        except Exception:
            self.db.rollback()
            raise

    def get_task(self, task_id: int) -> Optional[CrawlerTask]:
        return self.db.query(CrawlerTask).filter(CrawlerTask.id == task_id).first()

    def get_tasks(self, skip: int = 0, limit: int = 100, crawler_type: str = None, is_active: bool = None) -> List[CrawlerTask]:
        query = self.db.query(CrawlerTask)
        if crawler_type:
            query = query.filter(CrawlerTask.crawler_type == crawler_type)
        if is_active is not None:
            query = query.filter(CrawlerTask.is_active == is_active)
        return query.order_by(CrawlerTask.created_at.desc()).offset(skip).limit(limit).all()

    def update_task(self, task_id: int, update_data: dict) -> Optional[CrawlerTask]:
        try:
            task = self.get_task(task_id)
            if not task:
                return None
            
            for key, value in update_data.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            task.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(task)
            return task
        except Exception:
            self.db.rollback()
            raise

    def delete_task(self, task_id: int) -> bool:
        try:
            task = self.get_task(task_id)
            if not task:
                return False
            
            self.db.delete(task)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def run_task(self, task_id: int) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            return {"error": "任务不存在"}
        
        if not task.is_active:
            return {"error": "任务未激活"}
        
        try:
            task.status = "running"
            task.last_run_at = datetime.now()
            self.db.commit()
            
            return {
                "status": "running",
                "task_id": task_id,
                "message": "任务已启动"
            }
        except Exception:
            self.db.rollback()
            return {"error": "任务启动失败"}


class CrawlerLogService:
    def __init__(self, db: Session):
        self.db = db

    def create_log(self, log_data: dict) -> CrawlerLog:
        try:
            log = CrawlerLog(
                task_id=log_data.get("task_id"),
                log_level=log_data.get("log_level", "INFO"),
                message=log_data.get("message"),
                response_status=log_data.get("response_status"),
                records_count=log_data.get("records_count"),
                error_message=log_data.get("error_message"),
                created_at=datetime.now()
            )
            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)
            return log
        except Exception:
            self.db.rollback()
            raise

    def get_task_logs(self, task_id: int, skip: int = 0, limit: int = 100) -> List[CrawlerLog]:
        return self.db.query(CrawlerLog).filter(
            CrawlerLog.task_id == task_id
        ).order_by(CrawlerLog.created_at.desc()).offset(skip).limit(limit).all()

    def get_error_logs(self, skip: int = 0, limit: int = 100) -> List[CrawlerLog]:
        return self.db.query(CrawlerLog).filter(
            CrawlerLog.log_level.in_(["ERROR", "CRITICAL"])
        ).order_by(CrawlerLog.created_at.desc()).offset(skip).limit(limit).all()


class CrawlerDataService:
    def __init__(self, db: Session):
        self.db = db

    def create_data(self, data: dict) -> CrawlerData:
        try:
            crawler_data = CrawlerData(
                task_id=data.get("task_id"),
                data_type=data.get("data_type"),
                title=data.get("title"),
                content=data.get("content"),
                source_url=data.get("source_url"),
                publish_date=data.get("publish_date"),
                extra_data=data.get("extra_data"),
                created_at=datetime.now()
            )
            self.db.add(crawler_data)
            self.db.commit()
            self.db.refresh(crawler_data)
            return crawler_data
        except Exception:
            self.db.rollback()
            raise

    def get_data(self, data_id: int) -> Optional[CrawlerData]:
        return self.db.query(CrawlerData).filter(CrawlerData.id == data_id).first()

    def get_data_by_task(self, task_id: int, skip: int = 0, limit: int = 100) -> List[CrawlerData]:
        return self.db.query(CrawlerData).filter(
            CrawlerData.task_id == task_id
        ).order_by(CrawlerData.created_at.desc()).offset(skip).limit(limit).all()

    def get_data_by_type(self, data_type: str, skip: int = 0, limit: int = 100) -> List[CrawlerData]:
        return self.db.query(CrawlerData).filter(
            CrawlerData.data_type == data_type
        ).order_by(CrawlerData.created_at.desc()).offset(skip).limit(limit).all()
