"""
消费数据明细模型
对应数据库表：consumption_data
"""
from sqlalchemy import Column, String, Integer, DateTime, Float, Text
from datetime import datetime

from app.core.database import Base


class ConsumptionData(Base):
    """消费数据明细表"""
    __tablename__ = "consumption_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    data_id = Column(String(50), nullable=False, unique=True, index=True, comment="数据 ID")
    period = Column(String(20), nullable=False, comment="统计周期")
    year = Column(Integer, nullable=False, comment="年份")
    quarter = Column(Integer, nullable=True, comment="季度")
    month = Column(Integer, nullable=True, comment="月份")
    region = Column(String(100), nullable=False, comment="地区")
    consumption_type = Column(String(100), nullable=False, comment="消费类型")
    total_consumption = Column(Float, nullable=True, comment="总消费额")
    per_capita_consumption = Column(Float, nullable=True, comment="人均消费额")
    urban_consumption = Column(Float, nullable=True, comment="城镇消费额")
    rural_consumption = Column(Float, nullable=True, comment="农村消费额")
    retail_sales = Column(Float, nullable=True, comment="社会消费品零售总额")
    online_retail_sales = Column(Float, nullable=True, comment="网上零售额")
    catering_industry_sales = Column(Float, nullable=True, comment="餐饮业销售额")
    consumer_confidence_index = Column(Float, nullable=True, comment="消费者信心指数")
    cpi = Column(Float, nullable=True, comment="居民消费价格指数")
    ppi = Column(Float, nullable=True, comment="生产者价格指数")
    description = Column(Text, nullable=True, comment="数据描述")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
