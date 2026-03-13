"""
通知管理 Schema
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class NotificationType(str, Enum):
    """通知类型"""
    SYSTEM = "system"
    USER = "user"
    ALERT = "alert"
    TASK = "task"
    MESSAGE = "message"


class NotificationPriority(str, Enum):
    """通知优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationBase(BaseModel):
    """通知基础 Schema"""
    title: str = Field(..., min_length=1, max_length=200, description="通知标题")
    content: str = Field(..., min_length=1, description="通知内容")
    type: NotificationType = Field(default=NotificationType.SYSTEM, description="通知类型")
    priority: NotificationPriority = Field(default=NotificationPriority.NORMAL, description="优先级")
    action_url: Optional[str] = Field(None, max_length=500, description="操作链接")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="额外数据")


class NotificationCreate(NotificationBase):
    """创建通知 Schema"""
    user_id: Optional[int] = Field(None, description="用户 ID")


class NotificationUpdate(BaseModel):
    """更新通知 Schema"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    is_read: Optional[bool] = None


class NotificationResponse(NotificationBase):
    """通知响应 Schema"""
    id: int
    user_id: Optional[int]
    is_read: bool
    read_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class NotificationListResponse(BaseModel):
    """通知列表响应"""
    notifications: list[NotificationResponse]
    total: int
    page: int
    page_size: int


class UnreadCountResponse(BaseModel):
    """未读数量响应"""
    count: int


class NotificationSettingBase(BaseModel):
    """通知设置基础 Schema"""
    enable_email: bool = True
    enable_system: bool = True
    enable_sms: bool = False


class NotificationSettingCreate(NotificationSettingBase):
    """创建通知设置 Schema"""
    user_id: int


class NotificationSettingResponse(NotificationSettingBase):
    """通知设置响应"""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
