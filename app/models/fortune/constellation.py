"""星座数据模型

此模块定义星座数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class Constellation(Base):
    """星座数据模型"""
    
    __tablename__ = "constellations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), nullable=False, comment="星座名称")
    english_name = Column(String(50), nullable=True, comment="英文名")
    symbol = Column(String(10), nullable=True, comment="符号")
    
    start_date = Column(String(10), nullable=True, comment="开始日期")
    end_date = Column(String(10), nullable=True, comment="结束日期")
    
    element = Column(String(20), nullable=True, comment="元素")
    quality = Column(String(20), nullable=True, comment="性质")
    ruling_planet = Column(String(50), nullable=True, comment="守护星")
    
    strengths = Column(Text, nullable=True, comment="优点")
    weaknesses = Column(Text, nullable=True, comment="缺点")
    
    personality = Column(Text, nullable=True, comment="性格特点")
    love_style = Column(Text, nullable=True, comment="爱情观")
    career_style = Column(Text, nullable=True, comment="事业观")
    
    compatible_signs = Column(String(200), nullable=True, comment="相配星座")
    incompatible_signs = Column(String(200), nullable=True, comment="不相配星座")
    
    lucky_numbers = Column(String(50), nullable=True, comment="幸运数字")
    lucky_colors = Column(String(100), nullable=True, comment="幸运颜色")
    lucky_days = Column(String(100), nullable=True, comment="幸运日")
    lucky_directions = Column(String(100), nullable=True, comment="幸运方位")
    
    birthstone = Column(String(100), nullable=True, comment="诞生石")
    flower = Column(String(100), nullable=True, comment="幸运花")
    
    description = Column(Text, nullable=True, comment="描述")
    interpretation = Column(Text, nullable=True, comment="解读")
    
    source = Column(String(100), nullable=True, comment="来源")
    reference = Column(Text, nullable=True, comment="参考")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<Constellation(id={self.id}, name='{self.name}')>"
