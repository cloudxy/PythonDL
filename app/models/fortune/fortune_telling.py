"""运势数据模型

此模块定义运势数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class FortuneTelling(Base):
    """运势数据模型"""
    
    __tablename__ = "fortune_tellings"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="名称")
    category = Column(String(50), index=True, nullable=True, comment="分类")
    
    fortune_type = Column(String(50), nullable=True, comment="运势类型")
    period = Column(String(50), nullable=True, comment="周期")
    
    target_date = Column(Date, nullable=True, comment="目标日期")
    
    overall_score = Column(Numeric(5, 2), nullable=True, comment="综合评分")
    overall_fortune = Column(Text, nullable=True, comment="综合运势")
    
    career_score = Column(Numeric(5, 2), nullable=True, comment="事业评分")
    career_fortune = Column(Text, nullable=True, comment="事业运势")
    
    wealth_score = Column(Numeric(5, 2), nullable=True, comment="财运评分")
    wealth_fortune = Column(Text, nullable=True, comment="财运运势")
    
    love_score = Column(Numeric(5, 2), nullable=True, comment="感情评分")
    love_fortune = Column(Text, nullable=True, comment="感情运势")
    
    health_score = Column(Numeric(5, 2), nullable=True, comment="健康评分")
    health_fortune = Column(Text, nullable=True, comment="健康运势")
    
    study_score = Column(Numeric(5, 2), nullable=True, comment="学业评分")
    study_fortune = Column(Text, nullable=True, comment="学业运势")
    
    lucky_numbers = Column(String(50), nullable=True, comment="幸运数字")
    lucky_colors = Column(String(100), nullable=True, comment="幸运颜色")
    lucky_directions = Column(String(100), nullable=True, comment="幸运方位")
    
    good_for = Column(Text, nullable=True, comment="宜")
    bad_for = Column(Text, nullable=True, comment="忌")
    
    description = Column(Text, nullable=True, comment="描述")
    recommendations = Column(Text, nullable=True, comment="建议")
    
    source = Column(String(100), nullable=True, comment="来源")
    reference = Column(Text, nullable=True, comment="参考")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<FortuneTelling(id={self.id}, name='{self.name}', category='{self.category}')>"
