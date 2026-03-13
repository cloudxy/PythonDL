from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DashboardOverview(BaseModel):
    total_users: int
    active_users: int
    total_logs: int
    error_logs: int
    total_stocks: int


class UserStatistics(BaseModel):
    today_users: int
    week_users: int
    month_users: int


class LogStatistics(BaseModel):
    today_logs: int
    week_logs: int
    error_count: int


class SystemHealth(BaseModel):
    error_logs_24h: int
    error_rate: float
    health_status: str


class ModuleStatistic(BaseModel):
    module: str
    total_logs: int
    error_count: int


class RecentActivity(BaseModel):
    id: int
    log_level: str
    module: Optional[str]
    action: Optional[str]
    username: Optional[str]
    created_at: str
