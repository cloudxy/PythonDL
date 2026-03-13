from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, func, select
from app.models.weather import WeatherData, WeatherForecast
from datetime import datetime, timedelta
import json

from app.core.redis_client import redis_client


class WeatherService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.redis = redis_client
        self.cache_prefix = "weather:"
        self.cache_ttl = 600  # 10 分钟缓存

    async def create_weather_data(self, data: dict) -> WeatherData:
        try:
            weather = WeatherData(
                city_code=data.get("city_code"),
                city_name=data.get("city_name"),
                date=data.get("date") or data.get("weather_date"),
                weather=data.get("weather"),
                temperature=data.get("temperature"),
                humidity=data.get("humidity"),
                pressure=data.get("pressure"),
                wind_speed=data.get("wind_speed"),
                wind_direction=data.get("wind_direction"),
                weather_condition=data.get("weather_condition"),
                created_at=datetime.now()
            )
            self.db.add(weather)
            await self.db.commit()
            await self.db.refresh(weather)
            return weather
        except Exception:
            await self.db.rollback()
            raise

    async def get_weather_data(self, weather_id: int) -> Optional[WeatherData]:
        # 尝试从缓存获取
        cache_key = f"{self.cache_prefix}id:{weather_id}"
        cached = await self.redis.get(cache_key)
        if cached:
            return cached
        
        # 从数据库获取
        result = await self.db.execute(
            select(WeatherData).where(WeatherData.id == weather_id)
        )
        weather = result.scalar_one_or_none()
        
        # 写入缓存
        if weather:
            await self.redis.set(cache_key, weather.__dict__, expire=self.cache_ttl)
        
        return weather

    async def get_recent_weather(self, city_code: str, days: int = 30) -> List[WeatherData]:
        # 尝试从缓存获取
        cache_key = f"{self.cache_prefix}recent:{city_code}:{days}"
        cached = await self.redis.get(cache_key)
        if cached:
            return [WeatherData(**item) for item in cached]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        result = await self.db.execute(
            select(WeatherData)
            .where(WeatherData.city_code == city_code)
            .where(WeatherData.date >= start_date)
            .order_by(WeatherData.date.desc())
        )
        weathers = result.scalars().all()
        
        # 写入缓存
        weather_dicts = [w.__dict__ for w in weathers]
        await self.redis.set(cache_key, weather_dicts, expire=self.cache_ttl)
        
        return weathers

    async def get_weather_by_date(self, city_code: str, date: datetime) -> Optional[WeatherData]:
        # 尝试从缓存获取
        cache_key = f"{self.cache_prefix}date:{city_code}:{date.strftime('%Y-%m-%d')}"
        cached = await self.redis.get(cache_key)
        if cached:
            return WeatherData(**cached)
        
        result = await self.db.execute(
            select(WeatherData)
            .where(WeatherData.city_code == city_code)
            .where(WeatherData.date == date.date())
        )
        weather = result.scalar_one_or_none()
        
        # 写入缓存
        if weather:
            await self.redis.set(cache_key, weather.__dict__, expire=self.cache_ttl)
        
        return weather

    async def get_weather_data_paginated(
        self,
        city_code: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[WeatherData], int]:
        # 构建查询条件
        conditions = []
        if city_code:
            conditions.append(WeatherData.city_code.ilike(f"%{city_code}%"))
        if start_date:
            conditions.append(WeatherData.date >= start_date.date())
        if end_date:
            conditions.append(WeatherData.date <= end_date.date())
        
        # 查询总数
        if conditions:
            count_query = select(func.count()).select_from(WeatherData).where(and_(*conditions))
        else:
            count_query = select(func.count()).select_from(WeatherData)
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # 分页查询
        query = select(WeatherData)
        if conditions:
            query = query.where(and_(*conditions))
        query = query.order_by(WeatherData.date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.db.execute(query)
        weathers = result.scalars().all()
        
        return list(weathers), total

    def create_forecast(self, forecast_data: dict) -> WeatherForecast:
        try:
            forecast = WeatherForecast(
                city_code=forecast_data.get("city_code"),
                city_name=forecast_data.get("city_name"),
                forecast_date=forecast_data.get("forecast_date"),
                predicted_temp=forecast_data.get("predicted_temp"),
                predicted_humidity=forecast_data.get("predicted_humidity"),
                predicted_condition=forecast_data.get("predicted_condition"),
                confidence=forecast_data.get("confidence"),
                created_at=datetime.now()
            )
            self.db.add(forecast)
            self.db.commit()
            self.db.refresh(forecast)
            return forecast
        except Exception:
            self.db.rollback()
            raise

    def get_forecasts(self, city_code: str, days: int = 7) -> List[WeatherForecast]:
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days)
        return self.db.query(WeatherForecast).filter(
            WeatherForecast.city_code == city_code,
            WeatherForecast.forecast_date >= start_date,
            WeatherForecast.forecast_date <= end_date
        ).order_by(WeatherForecast.forecast_date).all()
