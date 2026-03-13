"""气象数据模型初始化

此模块导出所有气象相关的数据模型。
"""
from app.models.weather.weather_station import WeatherStation
from app.models.weather.weather import Weather
from app.models.weather.weather_forecast import WeatherForecast

# 向后兼容的别名
WeatherData = Weather

__all__ = [
    'WeatherStation',
    'Weather',
    'WeatherData',  # 别名
    'WeatherForecast',
]
