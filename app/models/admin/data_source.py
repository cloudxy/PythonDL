"""
数据源模型
对应数据库表：data_sources
"""
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.orm import relationship

from app.core.database import Base


class DataSource(Base):
    """数据源表"""
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    source_name = Column(String(100), nullable=False, unique=True, comment="数据源名称")
    description = Column(String(255), nullable=False, comment="数据源描述")
    total_records = Column(Integer, nullable=True, comment="总记录数")
    last_updated = Column(DateTime, nullable=True, comment="最后更新时间")
    created_at = Column(DateTime, nullable=False, comment="创建时间")
