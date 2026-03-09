"""周易数据模型

此模块定义周易数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class ZhouYi(Base):
    """周易数据模型"""
    
    __tablename__ = "zhou_yi"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hexagram_number = Column(Integer, index=True, nullable=False, comment="卦序")
    hexagram_name = Column(String(50), nullable=False, comment="卦名")
    
    upper_trigram = Column(String(20), nullable=True, comment="上卦")
    lower_trigram = Column(String(20), nullable=True, comment="下卦")
    
    binary_code = Column(String(10), nullable=True, comment="二进制码")
    
    image_description = Column(Text, nullable=True, comment="卦象描述")
    
    judgment = Column(Text, nullable=True, comment="卦辞")
    image = Column(Text, nullable=True, comment="象辞")
    
    line_texts = Column(Text, nullable=True, comment="爻辞(JSON)")
    
    meaning = Column(Text, nullable=True, comment="卦义")
    interpretation = Column(Text, nullable=True, comment="解读")
    
    fortune_career = Column(Text, nullable=True, comment="事业运势")
    fortune_wealth = Column(Text, nullable=True, comment="财运")
    fortune_love = Column(Text, nullable=True, comment="感情运势")
    fortune_health = Column(Text, nullable=True, comment="健康运势")
    
    good_for = Column(Text, nullable=True, comment="宜")
    bad_for = Column(Text, nullable=True, comment="忌")
    
    element = Column(String(20), nullable=True, comment="五行")
    direction = Column(String(20), nullable=True, comment="方位")
    
    description = Column(Text, nullable=True, comment="描述")
    recommendations = Column(Text, nullable=True, comment="建议")
    
    source = Column(String(100), nullable=True, comment="来源")
    reference = Column(Text, nullable=True, comment="参考")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<ZhouYi(id={self.id}, hexagram_number={self.hexagram_number}, hexagram_name='{self.hexagram_name}')>"
