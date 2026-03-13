"""
告警模型
对应数据库表：alerts
"""
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from app.core.database import Base


class Alert(Base):
    """告警表"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    alert_type = Column(String(20), nullable=False, index=True, comment="告警类型")
    message = Column(String(255), nullable=False, comment="告警消息")
    status = Column(String(20), nullable=False, index=True, comment="告警状态")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    resolved_at = Column(DateTime, nullable=True, comment="解决时间")
