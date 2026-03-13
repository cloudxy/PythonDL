from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.core.deps import get_db
from app.core.auth import get_current_user
from app.admin.services.dashboard_service import DashboardService
from app.schemas.admin.dashboard import (
    DashboardOverview, UserStatistics, LogStatistics, 
    SystemHealth, ModuleStatistic, RecentActivity
)

router = APIRouter(prefix="/dashboard", tags=["仪表盘"])


@router.get("/overview", response_model=DashboardOverview, summary="获取仪表盘概览")
def get_overview(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> DashboardOverview:
    service = DashboardService(db)
    return service.get_overview_stats()


@router.get("/users", response_model=UserStatistics, summary="获取用户统计")
def get_user_statistics(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> UserStatistics:
    service = DashboardService(db)
    return service.get_user_statistics()


@router.get("/logs", response_model=LogStatistics, summary="获取日志统计")
def get_log_statistics(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> LogStatistics:
    service = DashboardService(db)
    return service.get_log_statistics()


@router.get("/health", response_model=SystemHealth, summary="获取系统健康状态")
def get_system_health(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> SystemHealth:
    service = DashboardService(db)
    return service.get_system_health()


@router.get("/modules", response_model=List[ModuleStatistic], summary="获取模块统计")
def get_module_statistics(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[ModuleStatistic]:
    service = DashboardService(db)
    return service.get_module_statistics()


@router.get("/activities", response_model=List[RecentActivity], summary="获取最近活动")
def get_recent_activities(limit: int = 10, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[RecentActivity]:
    service = DashboardService(db)
    return service.get_recent_activities(limit=limit)
