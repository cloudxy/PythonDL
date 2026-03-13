"""
分析任务模型
对应数据库表：analyses
"""
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class Analysis(Base):
    """分析任务表"""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, comment="用户 ID")
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), index=True, comment="数据源 ID")
    analysis_type = Column(String(50), nullable=False, index=True, comment="分析类型")
    status = Column(String(20), nullable=False, comment="分析状态")
    result = Column(String(255), nullable=True, comment="分析结果")
    parameters = Column(Text, nullable=True, comment="分析参数")
    created_at = Column(DateTime, nullable=False, comment="创建时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")
    
    # 关系
    user = relationship("User", backref="analyses")
    data_source = relationship("DataSource", backref="analyses")
