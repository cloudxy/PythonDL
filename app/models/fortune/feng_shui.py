"""风水数据模型

此模块定义风水数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class FengShui(Base):
    """风水数据模型"""
    
    __tablename__ = "feng_shui"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="名称")
    category = Column(String(50), index=True, nullable=True, comment="分类")
    
    direction = Column(String(20), nullable=True, comment="方位")
    element = Column(String(20), nullable=True, comment="五行")
    trigram = Column(String(20), nullable=True, comment="八卦")
    
    lucky_numbers = Column(String(50), nullable=True, comment="幸运数字")
    lucky_colors = Column(String(100), nullable=True, comment="幸运颜色")
    lucky_directions = Column(String(100), nullable=True, comment="幸运方位")
    
    unlucky_numbers = Column(String(50), nullable=True, comment="忌讳数字")
    unlucky_colors = Column(String(100), nullable=True, comment="忌讳颜色")
    unlucky_directions = Column(String(100), nullable=True, comment="忌讳方位")
    
    description = Column(Text, nullable=True, comment="描述")
    interpretation = Column(Text, nullable=True, comment="解读")
    recommendations = Column(Text, nullable=True, comment="建议")
    
    source = Column(String(100), nullable=True, comment="来源")
    reference = Column(Text, nullable=True, comment="参考")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<FengShui(id={self.id}, name='{self.name}', category='{self.category}')>"
