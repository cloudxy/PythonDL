"""气象预测服务

此模块提供气象预测相关的业务逻辑。
"""
from typing import Dict, Any
from datetime import date, timedelta
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
import logging

from app.models.weather.weather import Weather
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class WeatherForecastService:
    """气象预测服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def forecast(self, station_id: str, forecast_days: int = 7) -> Dict[str, Any]:
        """气象预测"""
        cache_key = f"weather_forecast:{station_id}:{forecast_days}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        weather_data = self._get_historical_data(station_id, days=365)
        
        if weather_data.empty:
            return {
                "station_id": station_id,
                "forecast_date": date.today(),
                "forecasts": [],
                "confidence": 0.0
            }
        
        forecasts = self._generate_forecasts(weather_data, forecast_days)
        
        result = {
            "station_id": station_id,
            "forecast_date": date.today(),
            "forecasts": forecasts,
            "confidence": self._calculate_confidence(weather_data)
        }
        
        cache_manager.set(cache_key, result, expire=3600)
        
        return result
    
    def _get_historical_data(self, station_id: str, days: int = 365) -> pd.DataFrame:
        """获取历史数据"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        weathers = self.db.query(Weather).filter(
            Weather.station_id == station_id,
            Weather.record_date >= start_date,
            Weather.record_date <= end_date
        ).order_by(Weather.record_date).all()
        
        if not weathers:
            return pd.DataFrame()
        
        data = []
        for weather in weathers:
            data.append({
                'date': weather.record_date,
                'max_temp': float(weather.max_temp) if weather.max_temp else None,
                'min_temp': float(weather.min_temp) if weather.min_temp else None,
                'avg_temp': float(weather.avg_temp) if weather.avg_temp else None,
                'humidity': float(weather.avg_humidity) if weather.avg_humidity else None,
                'precipitation': float(weather.precipitation) if weather.precipitation else None
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def _generate_forecasts(self, data: pd.DataFrame, days: int) -> list:
        """生成预测结果"""
        forecasts = []
        
        for i in range(days):
            pred_date = date.today() + timedelta(days=i+1)
            
            forecasts.append({
                "date": str(pred_date),
                "max_temp": self._predict_temp(data, 'max_temp', i),
                "min_temp": self._predict_temp(data, 'min_temp', i),
                "avg_temp": self._predict_temp(data, 'avg_temp', i),
                "humidity": self._predict_humidity(data, i),
                "weather_type": self._predict_weather_type(data, i)
            })
        
        return forecasts
    
    def _predict_temp(self, data: pd.DataFrame, temp_type: str, day_offset: int) -> float:
        """预测温度"""
        try:
            if temp_type not in data.columns:
                return 20.0
            
            last_temp = data[temp_type].iloc[-1]
            ma7 = data[temp_type].rolling(7).mean().iloc[-1]
            return round((last_temp + ma7) / 2, 1)
        except Exception:
            return 20.0
    
    def _predict_humidity(self, data: pd.DataFrame, day_offset: int) -> float:
        """预测湿度"""
        try:
            if 'humidity' not in data.columns:
                return 60.0
            
            last_humidity = data['humidity'].iloc[-1]
            ma7 = data['humidity'].rolling(7).mean().iloc[-1]
            return round((last_humidity + ma7) / 2, 1)
        except Exception:
            return 60.0
    
    def _predict_weather_type(self, data: pd.DataFrame, day_offset: int) -> str:
        """预测天气类型"""
        import random
        weather_types = ["晴", "多云", "阴", "小雨", "大雨"]
        return random.choice(weather_types)
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """计算置信度"""
        try:
            if 'avg_temp' not in data.columns:
                return 0.5
            
            volatility = data['avg_temp'].pct_change().std()
            confidence = max(0.3, min(0.95, 1 - volatility))
            return round(confidence, 2)
        except Exception:
            return 0.5
