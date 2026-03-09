"""气象站点模型

此模块定义气象站点数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy import Index
from app.core.database import Base


class WeatherStation(Base):
    """气象站点模型"""
    
    __tablename__ = "weather_stations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    station_id = Column(String(20), unique=True, index=True, nullable=False, comment="站点ID")
    station_name = Column(String(100), nullable=False, comment="站点名称")
    province = Column(String(50), nullable=True, comment="省份")
    city = Column(String(50), nullable=True, comment="城市")
    district = Column(String(50), nullable=True, comment="区县")
    
    latitude = Column(Float, nullable=True, comment="纬度")
    longitude = Column(Float, nullable=True, comment="经度")
    altitude = Column(Float, nullable=True, comment="海拔高度")
    
    station_type = Column(String(50), nullable=True, comment="站点类型")
    is_active = Column(Integer, default=1, comment="是否活跃")
    description = Column(Text, nullable=True, comment="描述")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_weather_stations_province_city', 'province', 'city'),
    )
    
    def __repr__(self):
        return f"<WeatherStation(id={self.id}, station_name='{self.station_name}')>"
