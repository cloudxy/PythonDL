"""操作日志服务

此模块提供操作日志相关的业务逻辑。
"""
from typing import List, Optional
from datetime import datetime, date
from sqlalchemy.orm import Session

from app.models.admin.operation_log import OperationLog


class OperationLogService:
    """操作日志服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_log(self, log_id: int) -> Optional[OperationLog]:
        """获取日志"""
        return self.db.query(OperationLog).filter(OperationLog.id == log_id).first()
    
    def get_logs(
        self,
        skip: int = 0,
        limit: int = 20,
        user_id: Optional[int] = None,
        operation_type: Optional[str] = None
    ) -> List[OperationLog]:
        """获取日志列表"""
        query = self.db.query(OperationLog)
        
        if user_id:
            query = query.filter(OperationLog.user_id == user_id)
        
        if operation_type:
            query = query.filter(OperationLog.operation_type == operation_type)
        
        return query.order_by(OperationLog.created_at.desc()).offset(skip).limit(limit).all()
    
    def create_log(self, data: dict) -> OperationLog:
        """创建日志"""
        log = OperationLog(**data)
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log
    
    def count_today_logs(self) -> int:
        """统计今日日志数"""
        today = date.today()
        return self.db.query(OperationLog).filter(
            OperationLog.created_at >= datetime.combine(today, datetime.min.time())
        ).count()
    
    def get_recent_logs(self, limit: int = 10) -> List[OperationLog]:
        """获取最近日志"""
        return self.db.query(OperationLog).order_by(
            OperationLog.created_at.desc()
        ).limit(limit).all()
