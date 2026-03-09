"""面相数据模型

此模块定义面相数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class FaceReading(Base):
    """面相数据模型"""
    
    __tablename__ = "face_readings"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="名称")
    category = Column(String(50), index=True, nullable=True, comment="分类")
    
    face_part = Column(String(50), nullable=True, comment="面部部位")
    feature_type = Column(String(50), nullable=True, comment="特征类型")
    
    shape = Column(String(50), nullable=True, comment="形状")
    size = Column(String(50), nullable=True, comment="大小")
    position = Column(String(100), nullable=True, comment="位置")
    color = Column(String(50), nullable=True, comment="颜色")
    
    meaning = Column(Text, nullable=True, comment="含义")
    personality_traits = Column(Text, nullable=True, comment="性格特征")
    fortune_indication = Column(Text, nullable=True, comment="运势指示")
    
    good_signs = Column(Text, nullable=True, comment="吉相特征")
    bad_signs = Column(Text, nullable=True, comment="凶相特征")
    
    description = Column(Text, nullable=True, comment="描述")
    interpretation = Column(Text, nullable=True, comment="解读")
    recommendations = Column(Text, nullable=True, comment="建议")
    
    source = Column(String(100), nullable=True, comment="来源")
    reference = Column(Text, nullable=True, comment="参考")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def __repr__(self):
        return f"<FaceReading(id={self.id}, name='{self.name}', face_part='{self.face_part}')>"
