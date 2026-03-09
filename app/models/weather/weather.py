"""气象数据模型

此模块定义气象数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class Weather(Base):
    """气象数据模型"""
    
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    station_id = Column(String(20), index=True, nullable=False, comment="站点ID")
    record_date = Column(Date, index=True, nullable=False, comment="记录日期")
    
    max_temp = Column(Numeric(5, 2), nullable=True, comment="最高温度(℃)")
    min_temp = Column(Numeric(5, 2), nullable=True, comment="最低温度(℃)")
    avg_temp = Column(Numeric(5, 2), nullable=True, comment="平均温度(℃)")
    
    max_humidity = Column(Numeric(5, 2), nullable=True, comment="最高湿度(%)")
    min_humidity = Column(Numeric(5, 2), nullable=True, comment="最低湿度(%)")
    avg_humidity = Column(Numeric(5, 2), nullable=True, comment="平均湿度(%)")
    
    precipitation = Column(Numeric(10, 2), nullable=True, comment="降水量(mm)")
    evaporation = Column(Numeric(10, 2), nullable=True, comment="蒸发量(mm)")
    
    wind_speed = Column(Numeric(5, 2), nullable=True, comment="风速(m/s)")
    wind_direction = Column(String(20), nullable=True, comment="风向")
    max_wind_speed = Column(Numeric(5, 2), nullable=True, comment="最大风速(m/s)")
    
    pressure = Column(Numeric(10, 2), nullable=True, comment="气压(hPa)")
    visibility = Column(Numeric(10, 2), nullable=True, comment="能见度(km)")
    
    weather_type = Column(String(50), nullable=True, comment="天气类型")
    cloud_cover = Column(Numeric(5, 2), nullable=True, comment="云量(%)")
    
    uv_index = Column(Numeric(5, 2), nullable=True, comment="紫外线指数")
    air_quality_index = Column(Numeric(5, 2), nullable=True, comment="空气质量指数")
    
    sunrise = Column(String(10), nullable=True, comment="日出时间")
    sunset = Column(String(10), nullable=True, comment="日落时间")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_weather_data_station_date', 'station_id', 'record_date'),
        Index('idx_weather_data_date', 'record_date'),
    )
    
    def __repr__(self):
        return f"<Weather(id={self.id}, station_id='{self.station_id}', record_date='{self.record_date}')>"
