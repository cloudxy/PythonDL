# mypy: ignore-errors
# type: ignore
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from app.core.deps import get_db
from app.core.auth import get_current_user
from app.admin.services.system_log_service import SystemLogService
from app.schemas.admin.system_log import SystemLogCreate, SystemLogResponse

router = APIRouter(prefix="/logs", tags=["日志管理"])


@router.post("", response_model=SystemLogResponse, summary="创建系统日志")
async def create_log(log_in: SystemLogCreate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> SystemLogResponse:
    service = SystemLogService(db)
    return service.create_log(log_in.model_dump())


@router.get("", response_model=List[SystemLogResponse], summary="获取系统日志列表")
async def get_logs(skip: int = 0, limit: int = 100, log_level: Optional[str] = None, module: Optional[str] = None, 
             user_id: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, 
             db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[SystemLogResponse]:
    service = SystemLogService(db)
    return service.get_logs(skip=skip, limit=limit, log_level=log_level, module=module, 
                           user_id=user_id, start_date=start_date, end_date=end_date)


@router.get("/{log_id}", response_model=SystemLogResponse, summary="获取系统日志详情")
async def get_log(log_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> SystemLogResponse:
    service = SystemLogService(db)
    log = service.get_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="日志不存在")
    return log


@router.delete("/{log_id}", summary="删除系统日志")
async def delete_log(log_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = SystemLogService(db)
    success = service.delete_log(log_id)
    if not success:
        raise HTTPException(status_code=404, detail="日志不存在")
    return {"message": "删除成功"}


@router.delete("/cleanup/{days}", summary="清理旧日志")
async def delete_logs_older_than(days: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = SystemLogService(db)
    count = service.delete_logs_older_than(days)
    return {"message": f"已清理{count}条日志"}


@router.get("/errors/recent", response_model=List[SystemLogResponse], summary="获取错误日志")
async def get_error_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[SystemLogResponse]:
    service = SystemLogService(db)
    return service.get_error_logs(skip=skip, limit=limit)


@router.get("/user/{user_id}", response_model=List[SystemLogResponse], summary="获取用户操作日志")
async def get_user_logs(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[SystemLogResponse]:
    service = SystemLogService(db)
    return service.get_user_operation_logs(user_id, skip=skip, limit=limit)


@router.get("/module/{module}", response_model=List[SystemLogResponse], summary="获取模块日志")
async def get_module_logs(module: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[SystemLogResponse]:
    service = SystemLogService(db)
    return service.get_module_logs(module, skip=skip, limit=limit)
