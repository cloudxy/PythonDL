"""
面相记录模型
对应数据库表：face_readings
"""
from sqlalchemy import Column, String, Integer, DateTime, Float, Text
from datetime import datetime

from app.core.database import Base


class FaceReading(Base):
    """面相记录表"""
    __tablename__ = "face_readings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    face_id = Column(String(50), nullable=False, unique=True, index=True, comment="面相 ID")
    name = Column(String(100), nullable=False, comment="姓名")
    gender = Column(String(10), nullable=False, comment="性别")
    age = Column(Integer, nullable=False, comment="年龄")
    face_shape = Column(String(50), nullable=False, comment="脸型")
    forehead_type = Column(String(50), nullable=True, comment="额头类型")
    eyebrow_type = Column(String(50), nullable=True, comment="眉毛类型")
    eye_type = Column(String(50), nullable=True, comment="眼睛类型")
    nose_type = Column(String(50), nullable=True, comment="鼻子类型")
    mouth_type = Column(String(50), nullable=True, comment="嘴巴类型")
    chin_type = Column(String(50), nullable=True, comment="下巴类型")
    skin_type = Column(String(50), nullable=True, comment="皮肤类型")
    personality_analysis = Column(Text, nullable=True, comment="性格分析")
    career_analysis = Column(Text, nullable=True, comment="事业分析")
    relationship_analysis = Column(Text, nullable=True, comment="感情分析")
    health_analysis = Column(Text, nullable=True, comment="健康分析")
    wealth_analysis = Column(Text, nullable=True, comment="财富分析")
    luck_score = Column(Float, nullable=True, comment="运势评分")
    prediction_result = Column(Text, nullable=True, comment="预测结果")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
