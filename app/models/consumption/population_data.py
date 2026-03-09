"""人口数据模型

此模块定义人口数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric, BigInteger
from sqlalchemy import Index
from app.core.database import Base


class PopulationData(Base):
    """人口数据模型"""
    
    __tablename__ = "population_data"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region_code = Column(String(20), index=True, nullable=False, comment="区域代码")
    region_name = Column(String(100), nullable=False, comment="区域名称")
    region_level = Column(String(20), nullable=True, comment="区域级别")
    
    year = Column(Integer, index=True, nullable=False, comment="年份")
    
    total_population = Column(BigInteger, nullable=True, comment="总人口(人)")
    male_population = Column(BigInteger, nullable=True, comment="男性人口(人)")
    female_population = Column(BigInteger, nullable=True, comment="女性人口(人)")
    
    urban_population = Column(BigInteger, nullable=True, comment="城镇人口(人)")
    rural_population = Column(BigInteger, nullable=True, comment="乡村人口(人)")
    
    urbanization_rate = Column(Numeric(10, 3), nullable=True, comment="城镇化率(%)")
    
    birth_population = Column(Integer, nullable=True, comment="出生人口(人)")
    death_population = Column(Integer, nullable=True, comment="死亡人口(人)")
    
    birth_rate = Column(Numeric(10, 3), nullable=True, comment="出生率(‰)")
    death_rate = Column(Numeric(10, 3), nullable=True, comment="死亡率(‰)")
    natural_growth_rate = Column(Numeric(10, 3), nullable=True, comment="自然增长率(‰)")
    
    population_density = Column(Numeric(10, 2), nullable=True, comment="人口密度(人/平方公里)")
    
    age_0_14 = Column(BigInteger, nullable=True, comment="0-14岁人口(人)")
    age_15_64 = Column(BigInteger, nullable=True, comment="15-64岁人口(人)")
    age_65_plus = Column(BigInteger, nullable=True, comment="65岁以上人口(人)")
    
    dependency_ratio = Column(Numeric(10, 3), nullable=True, comment="抚养比(%)")
    elderly_ratio = Column(Numeric(10, 3), nullable=True, comment="老龄化率(%)")
    
    data_source = Column(String(100), nullable=True, comment="数据来源")
    data_quality = Column(String(20), default="official", comment="数据质量")
    
    description = Column(Text, nullable=True, comment="描述")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_population_data_region_year', 'region_code', 'year'),
    )
    
    def __repr__(self):
        return f"<PopulationData(id={self.id}, region_name='{self.region_name}', year={self.year})>"
