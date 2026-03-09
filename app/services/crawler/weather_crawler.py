"""气象数据爬虫

此模块提供气象数据采集功能。
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
import random

from app.models.weather.weather_station import WeatherStation
from app.models.weather.weather import Weather
from app.core.logger import get_logger

logger = get_logger("weather_crawler")


class WeatherCrawler:
    """气象数据爬虫类"""
    
    def __init__(self, db: Session):
        self.db = db
        self.status = {
            "is_running": False,
            "last_run": None,
            "total_records": 0,
            "error_count": 0
        }
    
    def crawl_weather_data(self, days: int = 365) -> Dict[str, Any]:
        """采集气象数据"""
        self.status["is_running"] = True
        start_time = datetime.now()
        
        try:
            logger.info(f"开始采集气象数据，天数: {days}")
            
            self._crawl_weather_stations()
            
            total_records = self._crawl_weather_records(days)
            
            self.status["is_running"] = False
            self.status["last_run"] = datetime.now()
            self.status["total_records"] = total_records
            
            logger.info(f"气象数据采集完成，共采集 {total_records} 条记录")
            
            return {
                "success": True,
                "total_records": total_records,
                "duration": (datetime.now() - start_time).seconds
            }
            
        except Exception as e:
            self.status["is_running"] = False
            self.status["error_count"] += 1
            logger.error(f"气象数据采集失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _crawl_weather_stations(self) -> int:
        """采集气象站点"""
        logger.info("开始采集气象站点")
        
        try:
            stations_data = self._generate_mock_stations()
            
            count = 0
            for data in stations_data:
                existing = self.db.query(WeatherStation).filter(
                    WeatherStation.station_id == data["station_id"]
                ).first()
                
                if not existing:
                    station = WeatherStation(**data)
                    self.db.add(station)
                    count += 1
            
            self.db.commit()
            logger.info(f"气象站点采集完成，新增 {count} 条记录")
            return count
            
        except Exception as e:
            logger.error(f"采集气象站点失败: {str(e)}")
            return 0
    
    def _crawl_weather_records(self, days: int) -> int:
        """采集气象记录"""
        logger.info(f"开始采集气象记录，天数: {days}")
        
        try:
            stations = self.db.query(WeatherStation).all()
            total_count = 0
            
            for station in stations:
                records = self._generate_mock_weather_records(station.station_id, days)
                
                for data in records:
                    existing = self.db.query(Weather).filter(
                        Weather.station_id == data["station_id"],
                        Weather.record_date == data["record_date"]
                    ).first()
                    
                    if not existing:
                        weather = Weather(**data)
                        self.db.add(weather)
                        total_count += 1
                
                if total_count % 100 == 0:
                    self.db.commit()
            
            self.db.commit()
            logger.info(f"气象记录采集完成，新增 {total_count} 条记录")
            return total_count
            
        except Exception as e:
            logger.error(f"采集气象记录失败: {str(e)}")
            return 0
    
    def _generate_mock_stations(self) -> List[Dict[str, Any]]:
        """生成模拟气象站点"""
        provinces = ["北京", "上海", "广东", "浙江", "江苏"]
        cities = {
            "北京": ["北京"],
            "上海": ["上海"],
            "广东": ["广州", "深圳", "东莞"],
            "浙江": ["杭州", "宁波", "温州"],
            "江苏": ["南京", "苏州", "无锡"]
        }
        
        stations = []
        station_id = 1000
        
        for province in provinces:
            for city in cities.get(province, []):
                stations.append({
                    "station_id": f"W{station_id}",
                    "station_name": f"{city}气象站",
                    "province": province,
                    "city": city,
                    "district": "市区",
                    "latitude": round(random.uniform(20, 45), 4),
                    "longitude": round(random.uniform(100, 130), 4),
                    "altitude": round(random.uniform(0, 500), 2),
                    "station_type": "自动站"
                })
                station_id += 1
        
        return stations
    
    def _generate_mock_weather_records(
        self,
        station_id: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """生成模拟气象记录"""
        from decimal import Decimal
        
        records = []
        base_temp = random.uniform(10, 30)
        
        for i in range(days):
            record_date = date.today() - timedelta(days=days-i)
            
            max_temp = base_temp + random.uniform(3, 8)
            min_temp = base_temp - random.uniform(3, 8)
            avg_temp = (max_temp + min_temp) / 2
            
            records.append({
                "station_id": station_id,
                "record_date": record_date,
                "max_temp": round(Decimal(max_temp), 2),
                "min_temp": round(Decimal(min_temp), 2),
                "avg_temp": round(Decimal(avg_temp), 2),
                "max_humidity": round(Decimal(random.uniform(70, 95)), 2),
                "min_humidity": round(Decimal(random.uniform(30, 60)), 2),
                "avg_humidity": round(Decimal(random.uniform(50, 80)), 2),
                "precipitation": round(Decimal(random.uniform(0, 50)), 2),
                "wind_speed": round(Decimal(random.uniform(1, 15)), 2),
                "wind_direction": random.choice(["北", "南", "东", "西", "东北", "西南"]),
                "weather_type": random.choice(["晴", "多云", "阴", "小雨", "大雨"]),
                "pressure": round(Decimal(random.uniform(1000, 1030)), 2)
            })
        
        return records
    
    def get_status(self) -> Dict[str, Any]:
        """获取爬虫状态"""
        return self.status
