"""气象预测模型

此模块定义气象预测数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class WeatherForecast(Base):
    """气象预测模型"""
    
    __tablename__ = "weather_forecasts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    station_id = Column(String(20), index=True, nullable=False, comment="站点ID")
    forecast_date = Column(Date, index=True, nullable=False, comment="预测日期")
    target_date = Column(Date, nullable=False, comment="目标日期")
    
    predicted_max_temp = Column(Numeric(5, 2), nullable=True, comment="预测最高温度(℃)")
    predicted_min_temp = Column(Numeric(5, 2), nullable=True, comment="预测最低温度(℃)")
    predicted_avg_temp = Column(Numeric(5, 2), nullable=True, comment="预测平均温度(℃)")
    
    predicted_humidity = Column(Numeric(5, 2), nullable=True, comment="预测湿度(%)")
    predicted_precipitation = Column(Numeric(10, 2), nullable=True, comment="预测降水量(mm)")
    predicted_wind_speed = Column(Numeric(5, 2), nullable=True, comment="预测风速(m/s)")
    predicted_wind_direction = Column(String(20), nullable=True, comment="预测风向")
    
    predicted_weather_type = Column(String(50), nullable=True, comment="预测天气类型")
    
    confidence = Column(Numeric(5, 3), nullable=True, comment="置信度")
    model_type = Column(String(50), nullable=True, comment="模型类型")
    model_version = Column(String(50), nullable=True, comment="模型版本")
    
    prediction_params = Column(Text, nullable=True, comment="预测参数")
    
    actual_max_temp = Column(Numeric(5, 2), nullable=True, comment="实际最高温度(℃)")
    actual_min_temp = Column(Numeric(5, 2), nullable=True, comment="实际最低温度(℃)")
    prediction_error = Column(Numeric(5, 2), nullable=True, comment="预测误差(℃)")
    
    status = Column(String(20), default="pending", comment="状态")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_weather_forecasts_station_date', 'station_id', 'forecast_date'),
    )
    
    def __repr__(self):
        return f"<WeatherForecast(id={self.id}, station_id='{self.station_id}', target_date='{self.target_date}')>"
