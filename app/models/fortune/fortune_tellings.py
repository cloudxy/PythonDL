"""
命理预测模型
对应数据库表：fortune_tellings
"""
from sqlalchemy import Column, String, Integer, DateTime, Float, Text
from datetime import datetime

from app.core.database import Base


class FortuneTelling(Base):
    """命理预测表"""
    __tablename__ = "fortune_tellings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    fortune_id = Column(String(50), nullable=False, unique=True, index=True, comment="命理 ID")
    name = Column(String(100), nullable=False, comment="姓名")
    gender = Column(String(10), nullable=False, comment="性别")
    birth_date = Column(DateTime, nullable=False, comment="出生日期")
    birth_time = Column(String(20), nullable=True, comment="出生时间")
    birth_place = Column(String(100), nullable=True, comment="出生地点")
    zodiac = Column(String(20), nullable=True, comment="生肖")
    constellation = Column(String(20), nullable=True, comment="星座")
    five_elements = Column(String(100), nullable=True, comment="五行分析")
    bazi = Column(String(200), nullable=True, comment="八字分析")
    overall_luck = Column(Text, nullable=True, comment="综合运势")
    career_luck = Column(Text, nullable=True, comment="事业运势")
    love_luck = Column(Text, nullable=True, comment="爱情运势")
    wealth_luck = Column(Text, nullable=True, comment="财富运势")
    health_luck = Column(Text, nullable=True, comment="健康运势")
    luck_score = Column(Float, nullable=True, comment="运势评分")
    lucky_direction = Column(String(100), nullable=True, comment="吉祥方位")
    lucky_colors = Column(String(100), nullable=True, comment="吉祥颜色")
    lucky_numbers = Column(String(100), nullable=True, comment="吉祥数字")
    prediction_result = Column(Text, nullable=True, comment="预测结果")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
