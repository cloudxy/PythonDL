"""系统配置模型

此模块定义系统配置数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from app.core.database import Base


class SystemConfig(Base):
    """系统配置模型"""
    
    __tablename__ = "system_configs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    config_key = Column(String(100), unique=True, index=True, nullable=False, comment="配置键")
    config_value = Column(Text, nullable=True, comment="配置值")
    config_type = Column(String(50), default="string", comment="配置类型")
    category = Column(String(50), default="system", comment="配置分类")
    description = Column(Text, nullable=True, comment="配置描述")
    is_public = Column(Boolean, default=False, comment="是否公开")
    is_active = Column(Boolean, default=True, comment="是否激活")
    sort_order = Column(Integer, default=0, comment="排序")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<SystemConfig(id={self.id}, config_key='{self.config_key}')>"
