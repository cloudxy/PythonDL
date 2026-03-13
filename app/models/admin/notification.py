"""
通知管理模块
"""
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base
from datetime import datetime


class NotificationType(str, enum.Enum):
    """通知类型"""
    SYSTEM = "system"  # 系统通知
    USER = "user"  # 用户通知
    ALERT = "alert"  # 预警通知
    TASK = "task"  # 任务通知
    MESSAGE = "message"  # 消息通知


class NotificationPriority(str, enum.Enum):
    """通知优先级"""
    LOW = "low"  # 低
    NORMAL = "normal"  # 普通
    HIGH = "high"  # 高
    URGENT = "urgent"  # 紧急


class Notification(Base):
    """通知表"""
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, comment="用户 ID")
    title = Column(String(200), nullable=False, comment="通知标题")
    content = Column(Text, nullable=False, comment="通知内容")
    type = Column(SQLEnum(NotificationType), nullable=False, default=NotificationType.SYSTEM, comment="通知类型")
    priority = Column(SQLEnum(NotificationPriority), nullable=False, default=NotificationPriority.NORMAL, comment="优先级")
    is_read = Column(Boolean, default=False, comment="是否已读")
    read_at = Column(DateTime(timezone=True), comment="阅读时间")
    action_url = Column(String(500), comment="操作链接")
    extra_data = Column(Text, comment="额外数据 (JSON)")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系
    user = relationship("User", back_populates="notifications")

    def mark_as_read(self):
        """标记为已读"""
        self.is_read = True
        self.read_at = func.now()


class NotificationSetting(Base):
    """通知设置表"""
    __tablename__ = "notification_settings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True, comment="用户 ID")
    enable_email = Column(Boolean, default=True, comment="启用邮件通知")
    enable_system = Column(Boolean, default=True, comment="启用系统通知")
    enable_sms = Column(Boolean, default=False, comment="启用短信通知")
    notification_types = Column(Text, comment="通知类型设置 (JSON)")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系
    user = relationship("User", back_populates="notification_settings")
