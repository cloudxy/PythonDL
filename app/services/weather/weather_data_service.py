"""气象数据服务层

此模块提供气象数据的数据库写入和管理功能。
"""
import asyncio
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.weather.weather import Weather
from app.models.weather.weather_station import WeatherStation
from app.core.logger import get_logger

logger = get_logger("weather_data_service")


class WeatherDataService:
    """气象数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_weather_station(self, station_data: Dict[str, Any]) -> Optional[WeatherStation]:
        """保存气象站点信息"""
        try:
            station_id = station_data.get("station_id", "")
            
            # 检查是否已存在
            stmt = select(WeatherStation).where(WeatherStation.station_id == station_id)
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新
                for key, value in station_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # 新增
                station = WeatherStation(**station_data)
                self.db.add(station)
                existing = station
            
            await self.db.commit()
            await self.db.refresh(existing)
            return existing
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存气象站点失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存气象站点异常：{e}")
            return None
    
    async def save_weather_record(self, weather_data: Dict[str, Any]) -> Optional[Weather]:
        """保存气象记录"""
        try:
            station_id = weather_data.get("station_id", "")
            record_date = weather_data.get("record_date", None)
            
            if not station_id or not record_date:
                logger.warning("缺少必要字段：station_id 或 record_date")
                return None
            
            # 检查是否已存在
            stmt = select(Weather).where(
                Weather.station_id == station_id,
                Weather.record_date == record_date
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新
                for key, value in weather_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # 新增
                weather = Weather(**weather_data)
                self.db.add(weather)
                existing = weather
            
            await self.db.commit()
            await self.db.refresh(existing)
            return existing
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存气象记录失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存气象记录异常：{e}")
            return None
    
    async def save_weather_records_batch(
        self,
        records: List[Dict[str, Any]]
    ) -> int:
        """批量保存气象记录"""
        if not records:
            return 0
        
        success_count = 0
        for record in records:
            result = await self.save_weather_record(record)
            if result:
                success_count += 1
        
        logger.info(f"批量保存气象记录：成功{success_count}/{len(records)}条")
        return success_count
    
    async def get_weather_records(
        self,
        station_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[Weather]:
        """获取气象记录"""
        try:
            stmt = select(Weather).where(Weather.station_id == station_id)
            
            if start_date:
                stmt = stmt.where(Weather.record_date >= start_date)
            if end_date:
                stmt = stmt.where(Weather.record_date <= end_date)
            
            stmt = stmt.order_by(Weather.record_date.desc()).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"获取气象记录失败：{e}")
            return []


async def save_weather_to_db(
    db: AsyncSession,
    station_data: Optional[Dict[str, Any]] = None,
    weather_data: Optional[Dict[str, Any]] = None
) -> bool:
    """保存气象数据到数据库的便捷函数"""
    service = WeatherDataService(db)
    
    if station_data:
        result = await service.save_weather_station(station_data)
        if not result:
            return False
    
    if weather_data:
        result = await service.save_weather_record(weather_data)
        if not result:
            return False
    
    return True


async def save_weather_batch_to_db(
    db: AsyncSession,
    stations: List[Dict[str, Any]],
    records: List[Dict[str, Any]]
) -> Dict[str, int]:
    """批量保存气象数据到数据库"""
    service = WeatherDataService(db)
    
    results = {
        "stations_saved": 0,
        "records_saved": 0
    }
    
    for station in stations:
        if await service.save_weather_station(station):
            results["stations_saved"] += 1
    
    results["records_saved"] = await service.save_weather_records_batch(records)
    
    return results
