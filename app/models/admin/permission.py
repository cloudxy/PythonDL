"""权限模型

此模块定义权限数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class Permission(Base):
    """权限模型"""
    
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    permission_name = Column(String(100), unique=True, index=True, nullable=False, comment="权限名称")
    permission_code = Column(String(100), unique=True, index=True, nullable=False, comment="权限编码")
    resource_type = Column(String(50), nullable=True, comment="资源类型")
    resource_path = Column(String(255), nullable=True, comment="资源路径")
    action = Column(String(50), nullable=True, comment="操作类型")
    description = Column(Text, nullable=True, comment="权限描述")
    parent_id = Column(Integer, nullable=True, comment="父权限ID")
    is_active = Column(Boolean, default=True, comment="是否激活")
    sort_order = Column(Integer, default=0, comment="排序")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    roles = relationship("RolePermission", back_populates="permission", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Permission(id={self.id}, permission_name='{self.permission_name}', permission_code='{self.permission_code}')>"
