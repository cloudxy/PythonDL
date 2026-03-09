"""角色权限关联模型

此模块定义角色权限关联数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base


class RolePermission(Base):
    """角色权限关联模型"""
    
    __tablename__ = "role_permissions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False, comment="角色ID")
    permission_id = Column(Integer, ForeignKey("permissions.id", ondelete="CASCADE"), nullable=False, comment="权限ID")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    
    role = relationship("Role", back_populates="permissions")
    permission = relationship("Permission", back_populates="roles")
    
    def __repr__(self):
        return f"<RolePermission(id={self.id}, role_id={self.role_id}, permission_id={self.permission_id})>"
