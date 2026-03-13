from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.models.user import User
from app.models.system import SystemLog, SystemConfig
from app.models.finance import Stock
from datetime import datetime, timedelta


class DashboardService:
    def __init__(self, db: Session):
        self.db = db

    def get_overview_stats(self) -> Dict[str, Any]:
        total_users = self.db.query(func.count(User.id)).scalar()
        active_users = self.db.query(func.count(User.id)).filter(User.is_active == True).scalar()
        total_logs = self.db.query(func.count(SystemLog.id)).scalar()
        error_logs = self.db.query(func.count(SystemLog.id)).filter(
            SystemLog.log_level.in_(["ERROR", "CRITICAL"])
        ).scalar()
        total_stocks = self.db.query(func.count(func.distinct(Stock.ts_code))).scalar()
        
        return {
            "total_users": total_users or 0,
            "active_users": active_users or 0,
            "total_logs": total_logs or 0,
            "error_logs": error_logs or 0,
            "total_stocks": total_stocks or 0
        }

    def get_user_statistics(self) -> Dict[str, Any]:
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        
        today_users = self.db.query(func.count(User.id)).filter(
            User.created_at >= today_start
        ).scalar()
        
        week_users = self.db.query(func.count(User.id)).filter(
            User.created_at >= week_start
        ).scalar()
        
        month_users = self.db.query(func.count(User.id)).filter(
            User.created_at >= month_start
        ).scalar()
        
        return {
            "today_users": today_users or 0,
            "week_users": week_users or 0,
            "month_users": month_users or 0
        }

    def get_log_statistics(self) -> Dict[str, Any]:
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        
        today_logs = self.db.query(func.count(SystemLog.id)).filter(
            SystemLog.created_at >= today_start
        ).scalar()
        
        week_logs = self.db.query(func.count(SystemLog.id)).filter(
            SystemLog.created_at >= week_start
        ).scalar()
        
        error_count = self.db.query(func.count(SystemLog.id)).filter(
            SystemLog.log_level.in_(["ERROR", "CRITICAL"]),
            SystemLog.created_at >= week_start
        ).scalar()
        
        return {
            "today_logs": today_logs or 0,
            "week_logs": week_logs or 0,
            "error_count": error_count or 0
        }

    def get_system_health(self) -> Dict[str, Any]:
        error_logs_24h = self.db.query(func.count(SystemLog.id)).filter(
            SystemLog.log_level.in_(["ERROR", "CRITICAL"]),
            SystemLog.created_at >= datetime.now() - timedelta(hours=24)
        ).scalar()
        
        error_rate = (error_logs_24h or 0) / max(1, self.db.query(func.count(SystemLog.id)).filter(
            SystemLog.created_at >= datetime.now() - timedelta(hours=24)
        ).scalar() or 1)
        
        if error_rate < 0.01:
            health_status = "healthy"
        elif error_rate < 0.05:
            health_status = "warning"
        else:
            health_status = "critical"
        
        return {
            "error_logs_24h": error_logs_24h or 0,
            "error_rate": round(error_rate, 4),
            "health_status": health_status
        }

    def get_module_statistics(self) -> List[Dict[str, Any]]:
        modules = self.db.query(
            SystemLog.module,
            func.count(SystemLog.id).label("total"),
            func.count(func.case((SystemLog.log_level.in_(["ERROR", "CRITICAL"]), 1))).label("errors")
        ).group_by(SystemLog.module).all()
        
        return [
            {
                "module": module.module or "unknown",
                "total_logs": module.total,
                "error_count": module.errors
            }
            for module in modules
        ]

    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        logs = self.db.query(SystemLog).order_by(
            SystemLog.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": log.id,
                "log_level": log.log_level,
                "module": log.module,
                "action": log.action,
                "username": log.username,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]
