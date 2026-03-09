"""GDP数据模型

此模块定义GDP数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric, BigInteger
from sqlalchemy import Index
from app.core.database import Base


class GDPData(Base):
    """GDP数据模型"""
    
    __tablename__ = "gdp_data"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region_code = Column(String(20), index=True, nullable=False, comment="区域代码")
    region_name = Column(String(100), nullable=False, comment="区域名称")
    region_level = Column(String(20), nullable=True, comment="区域级别")
    
    year = Column(Integer, index=True, nullable=False, comment="年份")
    quarter = Column(Integer, nullable=True, comment="季度")
    month = Column(Integer, nullable=True, comment="月份")
    
    gdp = Column(BigInteger, nullable=True, comment="GDP(万元)")
    gdp_growth = Column(Numeric(10, 3), nullable=True, comment="GDP增长率(%)")
    
    primary_industry = Column(BigInteger, nullable=True, comment="第一产业(万元)")
    secondary_industry = Column(BigInteger, nullable=True, comment="第二产业(万元)")
    tertiary_industry = Column(BigInteger, nullable=True, comment="第三产业(万元)")
    
    primary_ratio = Column(Numeric(10, 3), nullable=True, comment="第一产业占比(%)")
    secondary_ratio = Column(Numeric(10, 3), nullable=True, comment="第二产业占比(%)")
    tertiary_ratio = Column(Numeric(10, 3), nullable=True, comment="第三产业占比(%)")
    
    per_capita_gdp = Column(Numeric(15, 2), nullable=True, comment="人均GDP(元)")
    
    data_source = Column(String(100), nullable=True, comment="数据来源")
    data_quality = Column(String(20), default="official", comment="数据质量")
    
    description = Column(Text, nullable=True, comment="描述")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_gdp_data_region_year', 'region_code', 'year'),
    )
    
    def __repr__(self):
        return f"<GDPData(id={self.id}, region_name='{self.region_name}', year={self.year})>"
