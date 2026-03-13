"""
通知管理 API
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.admin.user import User
from app.schemas.admin.notification import (
    NotificationCreate,
    NotificationUpdate,
    NotificationResponse,
    NotificationListResponse,
    UnreadCountResponse,
    NotificationSettingResponse
)
from app.services.admin.notification_service import NotificationService
from app.models.admin.notification import NotificationType, NotificationPriority


router = APIRouter(prefix="/notifications", tags=["通知管理"])


@router.post("", response_model=NotificationResponse, summary="创建通知")
async def create_notification(
    notification: NotificationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建新通知"""
    service = NotificationService(db)
    
    # 如果是系统通知，需要管理员权限
    if notification.type == NotificationType.SYSTEM and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足")
    
    db_notification = await service.create_notification(notification)
    return db_notification


@router.get("/{notification_id}", response_model=NotificationResponse, summary="获取通知详情")
async def get_notification(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取通知详情"""
    service = NotificationService(db)
    notification = await service.get_notification(notification_id)
    
    if not notification:
        raise HTTPException(status_code=404, detail="通知不存在")
    
    # 检查权限
    if notification.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="无权查看此通知")
    
    return notification


@router.get("", response_model=NotificationListResponse, summary="获取通知列表")
async def get_notifications(
    is_read: Optional[bool] = Query(None, description="是否已读"),
    notification_type: Optional[NotificationType] = Query(None, description="通知类型"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户通知列表"""
    service = NotificationService(db)
    
    notifications, total = await service.get_user_notifications(
        user_id=current_user.id,
        is_read=is_read,
        notification_type=notification_type,
        page=page,
        page_size=page_size
    )
    
    return {
        "notifications": notifications,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.get("/unread/count", response_model=UnreadCountResponse, summary="获取未读通知数量")
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取未读通知数量"""
    service = NotificationService(db)
    count = await service.get_unread_count(current_user.id)
    return {"count": count}


@router.post("/{notification_id}/read", response_model=NotificationResponse, summary="标记为已读")
async def mark_as_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """标记通知为已读"""
    service = NotificationService(db)
    notification = await service.mark_as_read(notification_id, current_user.id)
    
    if not notification:
        raise HTTPException(status_code=404, detail="通知不存在")
    
    return notification


@router.post("/read-all", summary="标记所有为已读")
async def mark_all_as_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """标记所有通知为已读"""
    service = NotificationService(db)
    count = await service.mark_all_as_read(current_user.id)
    return {"message": f"已标记 {count} 条通知为已读"}


@router.delete("/{notification_id}", summary="删除通知")
async def delete_notification(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除通知"""
    service = NotificationService(db)
    success = await service.delete_notification(notification_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="通知不存在")
    
    return {"message": "删除成功"}


@router.delete("/read-all", summary="删除所有已读通知")
async def delete_all_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除所有已读通知"""
    service = NotificationService(db)
    count = await service.delete_all_read(current_user.id)
    return {"message": f"已删除 {count} 条已读通知"}


@router.get("/settings", response_model=NotificationSettingResponse, summary="获取通知设置")
async def get_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户通知设置"""
    service = NotificationService(db)
    settings = await service.get_settings(current_user.id)
    
    if not settings:
        # 创建默认设置
        settings = await service.update_settings(current_user.id, {
            "enable_email": True,
            "enable_system": True,
            "enable_sms": False
        })
    
    return settings


@router.put("/settings", response_model=NotificationSettingResponse, summary="更新通知设置")
async def update_settings(
    settings: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新用户通知设置"""
    service = NotificationService(db)
    updated_settings = await service.update_settings(current_user.id, settings)
    return updated_settings


@router.post("/system", response_model=NotificationResponse, summary="发送系统通知")
async def send_system_notification(
    title: str = Query(..., description="通知标题"),
    content: str = Query(..., description="通知内容"),
    priority: NotificationPriority = Query(NotificationPriority.NORMAL, description="优先级"),
    action_url: Optional[str] = Query(None, description="操作链接"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """发送系统通知（仅管理员）"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足")
    
    service = NotificationService(db)
    await service.send_system_notification(
        title=title,
        content=content,
        priority=priority,
        action_url=action_url
    )
    
    return {"message": "系统通知已发送"}
