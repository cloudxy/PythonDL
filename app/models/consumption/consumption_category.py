"""
消费类别模型
对应数据库表：consumption_categories
"""
from sqlalchemy import Column, String, Integer, DateTime, Float, Text
from datetime import datetime

from app.core.database import Base


class ConsumptionCategory(Base):
    """消费类别表"""
    __tablename__ = "consumption_categories"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category_code = Column(String(20), nullable=False, unique=True, index=True, comment="类别代码")
    category_name = Column(String(100), nullable=False, comment="类别名称")
    parent_code = Column(String(20), nullable=True, index=True, comment="父类别代码")
    level = Column(Integer, nullable=True, comment="类别层级")
    description = Column(Text, nullable=True, comment="类别描述")
    weight = Column(Float, nullable=True, comment="权重")
    is_active = Column(Integer, nullable=True, default=1, comment="是否启用 0-禁用 1-启用")
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
