"""
通知管理服务
"""
import json
from typing import List, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from app.models.admin.notification import Notification, NotificationType, NotificationPriority, NotificationSetting
from app.core.email import EmailService
from app.schemas.admin.notification import NotificationCreate, NotificationUpdate


class NotificationService:
    """通知服务类"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.email_service = EmailService()
    
    async def create_notification(
        self,
        notification: NotificationCreate,
        send_email: bool = False
    ) -> Notification:
        """创建通知"""
        db_notification = Notification(
            user_id=notification.user_id,
            title=notification.title,
            content=notification.content,
            type=notification.type,
            priority=notification.priority,
            action_url=notification.action_url,
            extra_data=json.dumps(notification.extra_data) if notification.extra_data else None
        )
        
        self.db.add(db_notification)
        await self.db.commit()
        await self.db.refresh(db_notification)
        
        # 发送邮件通知
        if send_email and notification.user_id:
            await self._send_notification_email(db_notification)
        
        return db_notification
    
    async def get_notification(self, notification_id: int) -> Optional[Notification]:
        """获取通知详情"""
        result = await self.db.execute(
            select(Notification)
            .options(selectinload(Notification.user))
            .where(Notification.id == notification_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_notifications(
        self,
        user_id: int,
        is_read: Optional[bool] = None,
        notification_type: Optional[NotificationType] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Notification], int]:
        """获取用户通知列表"""
        query = select(Notification).where(Notification.user_id == user_id)
        
        if is_read is not None:
            query = query.where(Notification.is_read == is_read)
        
        if notification_type:
            query = query.where(Notification.type == notification_type)
        
        # 获取总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar()
        
        # 分页
        query = query.order_by(Notification.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.db.execute(query)
        notifications = result.scalars().all()
        
        return list(notifications), total
    
    async def mark_as_read(self, notification_id: int, user_id: int) -> Optional[Notification]:
        """标记通知为已读"""
        notification = await self.get_notification(notification_id)
        
        if not notification or notification.user_id != user_id:
            return None
        
        notification.mark_as_read()
        await self.db.commit()
        await self.db.refresh(notification)
        
        return notification
    
    async def mark_all_as_read(self, user_id: int) -> int:
        """标记所有通知为已读"""
        result = await self.db.execute(
            update(Notification)
            .where(Notification.user_id == user_id)
            .where(Notification.is_read == False)
            .values(is_read=True, read_at=func.now())
        )
        await self.db.commit()
        return result.rowcount
    
    async def delete_notification(self, notification_id: int, user_id: int) -> bool:
        """删除通知"""
        notification = await self.get_notification(notification_id)
        
        if not notification or notification.user_id != user_id:
            return False
        
        await self.db.delete(notification)
        await self.db.commit()
        return True
    
    async def delete_all_read(self, user_id: int) -> int:
        """删除所有已读通知"""
        result = await self.db.execute(
            delete(Notification)
            .where(Notification.user_id == user_id)
            .where(Notification.is_read == True)
        )
        await self.db.commit()
        return result.rowcount
    
    async def get_unread_count(self, user_id: int) -> int:
        """获取未读通知数量"""
        result = await self.db.execute(
            select(func.count())
            .select_from(Notification)
            .where(Notification.user_id == user_id)
            .where(Notification.is_read == False)
        )
        return result.scalar() or 0
    
    async def get_settings(self, user_id: int) -> Optional[NotificationSetting]:
        """获取用户通知设置"""
        result = await self.db.execute(
            select(NotificationSetting)
            .where(NotificationSetting.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def update_settings(
        self,
        user_id: int,
        settings: dict
    ) -> NotificationSetting:
        """更新用户通知设置"""
        db_setting = await self.get_settings(user_id)
        
        if not db_setting:
            db_setting = NotificationSetting(user_id=user_id)
            self.db.add(db_setting)
        
        # 更新设置
        for key, value in settings.items():
            if hasattr(db_setting, key):
                setattr(db_setting, key, value)
        
        await self.db.commit()
        await self.db.refresh(db_setting)
        return db_setting
    
    async def _send_notification_email(self, notification: Notification):
        """发送通知邮件"""
        if not notification.user:
            return
        
        # 获取用户设置
        settings = await self.get_settings(notification.user_id)
        if not settings or not settings.enable_email:
            return
        
        # 发送邮件
        await self.email_service.send_password_reset(
            to_email=notification.user.email,
            username=notification.user.username,
            reset_token=notification.id  # 仅作为示例
        )
    
    async def send_system_notification(
        self,
        title: str,
        content: str,
        user_id: Optional[int] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        action_url: Optional[str] = None,
        extra_data: Optional[dict] = None
    ):
        """发送系统通知"""
        notification = NotificationCreate(
            user_id=user_id,
            title=title,
            content=content,
            type=NotificationType.SYSTEM,
            priority=priority,
            action_url=action_url,
            extra_data=extra_data
        )
        await self.create_notification(notification)
    
    async def send_alert(
        self,
        user_id: int,
        title: str,
        content: str,
        priority: NotificationPriority = NotificationPriority.HIGH,
        action_url: Optional[str] = None
    ):
        """发送预警通知"""
        notification = NotificationCreate(
            user_id=user_id,
            title=title,
            content=content,
            type=NotificationType.ALERT,
            priority=priority,
            action_url=action_url
        )
        await self.create_notification(notification, send_email=True)
