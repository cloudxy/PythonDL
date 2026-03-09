"""八字数据模型

此模块定义八字数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class Bazi(Base):
    """八字数据模型"""
    
    __tablename__ = "bazi"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="名称")
    
    year_pillar = Column(String(20), nullable=True, comment="年柱")
    month_pillar = Column(String(20), nullable=True, comment="月柱")
    day_pillar = Column(String(20), nullable=True, comment="日柱")
    hour_pillar = Column(String(20), nullable=True, comment="时柱")
    
    year_stem = Column(String(10), nullable=True, comment="年干")
    year_branch = Column(String(10), nullable=True, comment="年支")
    month_stem = Column(String(10), nullable=True, comment="月干")
    month_branch = Column(String(10), nullable=True, comment="月支")
    day_stem = Column(String(10), nullable=True, comment="日干")
    day_branch = Column(String(10), nullable=True, comment="日支")
    hour_stem = Column(String(10), nullable=True, comment="时干")
    hour_branch = Column(String(10), nullable=True, comment="时支")
    
    day_master = Column(String(20), nullable=True, comment="日主")
    day_master_element = Column(String(20), nullable=True, comment="日主五行")
    
    five_elements = Column(Text, nullable=True, comment="五行分布")
    ten_gods = Column(Text, nullable=True, comment="十神")
    
    life_analysis = Column(Text, nullable=True, comment="命理分析")
    personality = Column(Text, nullable=True, comment="性格特点")
    career = Column(Text, nullable=True, comment="事业运势")
    wealth = Column(Text, nullable=True, comment="财运")
    marriage = Column(Text, nullable=True, comment="婚姻运势")
    health = Column(Text, nullable=True, comment="健康运势")
    
    lucky_elements = Column(String(100), nullable=True, comment="喜用神")
    unlucky_elements = Column(String(100), nullable=True, comment="忌神")
    
    description = Column(Text, nullable=True, comment="描述")
    interpretation = Column(Text, nullable=True, comment="解读")
    recommendations = Column(Text, nullable=True, comment="建议")
    
    source = Column(String(100), nullable=True, comment="来源")
    reference = Column(Text, nullable=True, comment="参考")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<Bazi(id={self.id}, name='{self.name}')>"
