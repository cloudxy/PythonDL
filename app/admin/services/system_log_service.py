# type: ignore
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.system import SystemLog
from datetime import datetime, timedelta


class SystemLogService:
    def __init__(self, db: Session):
        self.db = db

    def create_log(self, log_data: dict) -> SystemLog:
        try:
            log = SystemLog(
                log_level=log_data.get("log_level", "INFO"),
                module=log_data.get("module"),
                action=log_data.get("action"),
                user_id=log_data.get("user_id"),
                username=log_data.get("username"),
                request_method=log_data.get("request_method"),
                request_url=log_data.get("request_url"),
                request_params=log_data.get("request_params"),
                response_status=log_data.get("response_status"),
                error_message=log_data.get("error_message"),
                ip_address=log_data.get("ip_address"),
                user_agent=log_data.get("user_agent"),
                created_at=datetime.now()
            )
            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)
            return log
        except Exception:
            self.db.rollback()
            raise

    def get_log(self, log_id: int) -> Optional[SystemLog]:
        return self.db.query(SystemLog).filter(SystemLog.id == log_id).first()

    def get_logs(self, skip: int = 0, limit: int = 100, log_level: str = None, 
                 module: str = None, user_id: int = None, start_date: datetime = None, 
                 end_date: datetime = None) -> List[SystemLog]:
        query = self.db.query(SystemLog)
        
        if log_level:
            query = query.filter(SystemLog.log_level == log_level)
        if module:
            query = query.filter(SystemLog.module == module)
        if user_id:
            query = query.filter(SystemLog.user_id == user_id)
        if start_date:
            query = query.filter(SystemLog.created_at >= start_date)
        if end_date:
            query = query.filter(SystemLog.created_at <= end_date)
        
        return query.order_by(SystemLog.created_at.desc()).offset(skip).limit(limit).all()

    def delete_log(self, log_id: int) -> bool:
        try:
            log = self.get_log(log_id)
            if not log:
                return False
            
            self.db.delete(log)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def delete_logs_older_than(self, days: int) -> int:
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            result = self.db.query(SystemLog).filter(
                SystemLog.created_at < cutoff_date
            ).delete()
            self.db.commit()
            return result
        except Exception:
            self.db.rollback()
            raise

    def get_error_logs(self, skip: int = 0, limit: int = 100) -> List[SystemLog]:
        return self.db.query(SystemLog).filter(
            SystemLog.log_level.in_(["ERROR", "CRITICAL"])
        ).order_by(SystemLog.created_at.desc()).offset(skip).limit(limit).all()

    def get_user_operation_logs(self, user_id: int, skip: int = 0, limit: int = 100) -> List[SystemLog]:
        return self.db.query(SystemLog).filter(
            SystemLog.user_id == user_id
        ).order_by(SystemLog.created_at.desc()).offset(skip).limit(limit).all()

    def get_module_logs(self, module: str, skip: int = 0, limit: int = 100) -> List[SystemLog]:
        return self.db.query(SystemLog).filter(
            SystemLog.module == module
        ).order_by(SystemLog.created_at.desc()).offset(skip).limit(limit).all()
