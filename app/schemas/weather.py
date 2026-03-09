"""气象分析相关Schema

此模块定义气象分析相关的数据验证模型。
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal


class WeatherStationCreate(BaseModel):
    """气象站点创建"""
    station_id: str = Field(..., max_length=20)
    station_name: str = Field(..., max_length=100)
    province: Optional[str] = Field(None, max_length=50)
    city: Optional[str] = Field(None, max_length=50)
    district: Optional[str] = Field(None, max_length=50)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    station_type: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None


class WeatherStationResponse(BaseModel):
    """气象站点响应"""
    id: int
    station_id: str
    station_name: str
    province: Optional[str]
    city: Optional[str]
    district: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    station_type: Optional[str]
    is_active: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class WeatherDataCreate(BaseModel):
    """气象数据创建"""
    station_id: str = Field(..., max_length=20)
    record_date: date
    max_temp: Optional[Decimal] = None
    min_temp: Optional[Decimal] = None
    avg_temp: Optional[Decimal] = None
    max_humidity: Optional[Decimal] = None
    min_humidity: Optional[Decimal] = None
    avg_humidity: Optional[Decimal] = None
    precipitation: Optional[Decimal] = None
    wind_speed: Optional[Decimal] = None
    wind_direction: Optional[str] = Field(None, max_length=20)
    weather_type: Optional[str] = Field(None, max_length=50)


class WeatherDataResponse(BaseModel):
    """气象数据响应"""
    id: int
    station_id: str
    record_date: date
    max_temp: Optional[Decimal]
    min_temp: Optional[Decimal]
    avg_temp: Optional[Decimal]
    max_humidity: Optional[Decimal]
    min_humidity: Optional[Decimal]
    avg_humidity: Optional[Decimal]
    precipitation: Optional[Decimal]
    wind_speed: Optional[Decimal]
    wind_direction: Optional[str]
    weather_type: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class WeatherForecastRequest(BaseModel):
    """气象预测请求"""
    station_id: str = Field(..., max_length=20)
    forecast_days: int = Field(default=7, ge=1, le=15)


class WeatherForecastResponse(BaseModel):
    """气象预测响应"""
    station_id: str
    forecast_date: date
    forecasts: List[dict]
    confidence: float
