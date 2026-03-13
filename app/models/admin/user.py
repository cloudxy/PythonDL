"""用户模型

此模块定义用户数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


class User(Base):
    """用户模型"""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    email = Column(String(100), unique=True, index=True, nullable=False, comment="邮箱")
    password = Column(String(100), nullable=False, comment="密码")
    real_name = Column(String(50), nullable=True, comment="真实姓名")
    phone = Column(String(20), nullable=True, comment="手机号")
    avatar = Column(String(255), nullable=True, comment="头像 URL")
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=True, comment="角色 ID")
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_superuser = Column(Boolean, default=False, comment="是否超级管理员")
    last_login = Column(DateTime, nullable=True, comment="最后登录时间")
    login_count = Column(Integer, default=0, comment="登录次数")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    role = relationship("Role", back_populates="users")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    notification_settings = relationship("NotificationSetting", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
    
    @property
    def is_authenticated(self) -> bool:
        return True
    
    @property
    def is_anonymous(self) -> bool:
        return False
