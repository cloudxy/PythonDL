"""爬虫任务管理 API

此模块提供爬虫任务管理相关接口。
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.database import get_db
from app.core.response import success_response, error_response
from app.models.admin.crawler_task import CrawlerTask, CrawlerLog, TaskStatus
from app.schemas.admin.crawler_task import (
    CrawlerTaskCreate, CrawlerTaskUpdate, CrawlerTaskResponse,
    CrawlerTaskList, CrawlerLogResponse, CrawlerLogList
)
from app.services.admin.crawler_task_service import CrawlerTaskService

router = APIRouter(prefix="/crawler-tasks", tags=["爬虫任务管理"])


@router.get("", response_model=CrawlerTaskList)
async def list_crawler_tasks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    task_type: Optional[str] = Query(None, description="任务类型"),
    status: Optional[str] = Query(None, description="任务状态"),
    is_active: Optional[bool] = Query(None, description="是否启用"),
    db: AsyncSession = Depends(get_db)
):
    """获取爬虫任务列表"""
    service = CrawlerTaskService(db)
    
    tasks, total = await service.list_tasks(
        page=page,
        page_size=page_size,
        task_type=task_type,
        status=status,
        is_active=is_active
    )
    
    return success_response({
        "items": tasks,
        "total": total,
        "page": page,
        "page_size": page_size
    })


@router.get("/{task_id}", response_model=CrawlerTaskResponse)
async def get_crawler_task(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """获取爬虫任务详情"""
    service = CrawlerTaskService(db)
    task = await service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return success_response(task)


@router.post("", response_model=CrawlerTaskResponse)
async def create_crawler_task(
    task_data: CrawlerTaskCreate,
    db: AsyncSession = Depends(get_db)
):
    """创建爬虫任务"""
    service = CrawlerTaskService(db)
    task = await service.create_task(task_data)
    
    return success_response(task)


@router.put("/{task_id}", response_model=CrawlerTaskResponse)
async def update_crawler_task(
    task_id: int,
    task_data: CrawlerTaskUpdate,
    db: AsyncSession = Depends(get_db)
):
    """更新爬虫任务"""
    service = CrawlerTaskService(db)
    task = await service.update_task(task_id, task_data)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return success_response(task)


@router.delete("/{task_id}")
async def delete_crawler_task(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """删除爬虫任务"""
    service = CrawlerTaskService(db)
    success = await service.delete_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return success_response({"message": "删除成功"})


@router.post("/{task_id}/start")
async def start_crawler_task(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """启动爬虫任务"""
    service = CrawlerTaskService(db)
    task = await service.start_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return success_response({"message": "任务已启动", "task_id": task_id})


@router.post("/{task_id}/stop")
async def stop_crawler_task(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """停止爬虫任务"""
    service = CrawlerTaskService(db)
    task = await service.stop_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return success_response({"message": "任务已停止", "task_id": task_id})


@router.get("/{task_id}/logs", response_model=CrawlerLogList)
async def list_crawler_logs(
    task_id: int,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    log_level: Optional[str] = Query(None, description="日志级别"),
    db: AsyncSession = Depends(get_db)
):
    """获取爬虫任务日志列表"""
    service = CrawlerTaskService(db)
    
    logs, total = await service.list_logs(
        task_id=task_id,
        page=page,
        page_size=page_size,
        log_level=log_level
    )
    
    return success_response({
        "items": logs,
        "total": total,
        "page": page,
        "page_size": page_size
    })


@router.get("/logs/{log_id}", response_model=CrawlerLogResponse)
async def get_crawler_log(
    log_id: int,
    db: AsyncSession = Depends(get_db)
):
    """获取日志详情"""
    service = CrawlerTaskService(db)
    log = await service.get_log(log_id)
    
    if not log:
        raise HTTPException(status_code=404, detail="日志不存在")
    
    return success_response(log)


@router.post("/{task_id}/run-now")
async def run_crawler_task_now(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """立即执行爬虫任务"""
    service = CrawlerTaskService(db)
    execution_id = await service.run_task_now(task_id)
    
    return success_response({
        "message": "任务执行中",
        "task_id": task_id,
        "execution_id": execution_id
    })


@router.get("/{task_id}/status")
async def get_crawler_task_status(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """获取任务状态"""
    service = CrawlerTaskService(db)
    status = await service.get_task_status(task_id)
    
    return success_response(status)
