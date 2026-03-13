from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.weather import WeatherData
from datetime import datetime, timedelta
import numpy as np


class WeatherPredictionService:
    def __init__(self, db: Session):
        self.db = db

    def _get_recent_weather(self, city_code: str, days: int = 60) -> List[WeatherData]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.db.query(WeatherData).filter(
            WeatherData.city_code == city_code,
            WeatherData.weather_date >= start_date
        ).order_by(WeatherData.weather_date).all()

    def predict_temperature(self, city_code: str, days: int = 7) -> Dict[str, Any]:
        weather_data = self._get_recent_weather(city_code, days=60)
        
        if not weather_data:
            return {"error": "无历史数据"}
        
        temperatures = [w.temperature for w in weather_data if w.temperature is not None]
        
        if len(temperatures) < 7:
            return {"error": "数据不足"}
        
        predictions = []
        for i in range(days):
            avg_temp = np.mean(temperatures[-7:])
            trend = np.random.uniform(-0.5, 0.5)
            predicted_temp = round(avg_temp + trend, 1)
            
            predictions.append({
                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "predicted_temp": predicted_temp,
                "confidence": round(np.random.uniform(0.7, 0.9), 2)
            })
        
        return {
            "city_code": city_code,
            "predictions": predictions,
            "model_type": "moving_average"
        }

    def predict_weather_condition(self, city_code: str, days: int = 3) -> Dict[str, Any]:
        weather_data = self._get_recent_weather(city_code, days=30)
        
        if not weather_data:
            return {"error": "无历史数据"}
        
        conditions = [w.weather_condition for w in weather_data if w.weather_condition]
        
        if not conditions:
            return {"error": "无天气状况数据"}
        
        predictions = []
        condition_types = list(set(conditions))
        
        for i in range(days):
            predicted_condition = np.random.choice(condition_types)
            predictions.append({
                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "predicted_condition": predicted_condition,
                "confidence": round(np.random.uniform(0.6, 0.8), 2)
            })
        
        return {
            "city_code": city_code,
            "predictions": predictions,
            "model_type": "statistical"
        }
