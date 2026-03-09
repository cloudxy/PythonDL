"""气象服务

此模块提供气象相关的业务逻辑。
"""
from typing import List, Optional
from datetime import date
from sqlalchemy.orm import Session

from app.models.weather.weather_station import WeatherStation
from app.models.weather.weather import Weather


class WeatherService:
    """气象服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_station(self, station_id: int) -> Optional[WeatherStation]:
        """获取气象站点"""
        return self.db.query(WeatherStation).filter(WeatherStation.id == station_id).first()
    
    def get_station_by_code(self, station_id: str) -> Optional[WeatherStation]:
        """通过站点ID获取气象站点"""
        return self.db.query(WeatherStation).filter(WeatherStation.station_id == station_id).first()
    
    def get_stations(
        self,
        skip: int = 0,
        limit: int = 20,
        province: Optional[str] = None,
        city: Optional[str] = None
    ) -> List[WeatherStation]:
        """获取气象站点列表"""
        query = self.db.query(WeatherStation)
        
        if province:
            query = query.filter(WeatherStation.province == province)
        
        if city:
            query = query.filter(WeatherStation.city == city)
        
        return query.offset(skip).limit(limit).all()
    
    def create_station(self, data: dict) -> WeatherStation:
        """创建气象站点"""
        station = WeatherStation(**data)
        self.db.add(station)
        self.db.commit()
        self.db.refresh(station)
        return station
    
    def update_station(self, station_id: int, data: dict) -> Optional[WeatherStation]:
        """更新气象站点"""
        station = self.get_station(station_id)
        if not station:
            return None
        
        for key, value in data.items():
            if hasattr(station, key) and value is not None:
                setattr(station, key, value)
        
        self.db.commit()
        self.db.refresh(station)
        return station
    
    def delete_station(self, station_id: int) -> bool:
        """删除气象站点"""
        station = self.get_station(station_id)
        if not station:
            return False
        
        self.db.delete(station)
        self.db.commit()
        return True
    
    def get_weather_data(
        self,
        skip: int = 0,
        limit: int = 20,
        station_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Weather]:
        """获取气象数据"""
        query = self.db.query(Weather)
        
        if station_id:
            query = query.filter(Weather.station_id == station_id)
        
        if start_date:
            query = query.filter(Weather.record_date >= start_date)
        
        if end_date:
            query = query.filter(Weather.record_date <= end_date)
        
        return query.order_by(Weather.record_date.desc()).offset(skip).limit(limit).all()
    
    def create_weather_data(self, data: dict) -> Weather:
        """创建气象数据"""
        weather = Weather(**data)
        self.db.add(weather)
        self.db.commit()
        self.db.refresh(weather)
        return weather
