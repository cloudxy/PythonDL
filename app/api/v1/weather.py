"""气象分析API路由

此模块定义气象分析相关的API接口。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.schemas.weather import (
    WeatherStationCreate,
    WeatherStationResponse,
    WeatherDataCreate,
    WeatherDataResponse,
    WeatherForecastRequest,
    WeatherForecastResponse
)
from app.services.weather.weather_service import WeatherService
from app.services.weather.weather_forecast_service import WeatherForecastService

router = APIRouter()


@router.get("/stations", response_model=List[WeatherStationResponse])
async def list_stations(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    province: Optional[str] = None,
    city: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取气象站点列表"""
    weather_service = WeatherService(db)
    stations = weather_service.get_stations(
        skip=skip,
        limit=limit,
        province=province,
        city=city
    )
    return [WeatherStationResponse.from_orm(s) for s in stations]


@router.post("/stations", response_model=WeatherStationResponse)
async def create_station(
    station_data: WeatherStationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("weather:create"))
):
    """创建气象站点"""
    weather_service = WeatherService(db)
    station = weather_service.create_station(station_data.dict())
    return WeatherStationResponse.from_orm(station)


@router.get("/stations/{station_id}", response_model=WeatherStationResponse)
async def get_station(
    station_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取气象站点详情"""
    weather_service = WeatherService(db)
    station = weather_service.get_station(station_id)
    if not station:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="站点不存在")
    return WeatherStationResponse.from_orm(station)


@router.put("/stations/{station_id}", response_model=WeatherStationResponse)
async def update_station(
    station_id: int,
    station_data: WeatherStationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("weather:update"))
):
    """更新气象站点"""
    weather_service = WeatherService(db)
    station = weather_service.update_station(
        station_id,
        station_data.dict(exclude_unset=True)
    )
    return WeatherStationResponse.from_orm(station)


@router.delete("/stations/{station_id}")
async def delete_station(
    station_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("weather:delete"))
):
    """删除气象站点"""
    weather_service = WeatherService(db)
    weather_service.delete_station(station_id)
    return {"message": "删除成功"}


@router.get("/data", response_model=List[WeatherDataResponse])
async def list_weather_data(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    station_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取气象数据列表"""
    weather_service = WeatherService(db)
    data = weather_service.get_weather_data(
        skip=skip,
        limit=limit,
        station_id=station_id,
        start_date=start_date,
        end_date=end_date
    )
    return [WeatherDataResponse.from_orm(d) for d in data]


@router.post("/data", response_model=WeatherDataResponse)
async def create_weather_data(
    data: WeatherDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("weather:create"))
):
    """创建气象数据"""
    weather_service = WeatherService(db)
    weather_data = weather_service.create_weather_data(data.dict())
    return WeatherDataResponse.from_orm(weather_data)


@router.post("/forecast", response_model=WeatherForecastResponse)
async def forecast_weather(
    forecast_request: WeatherForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("weather:forecast"))
):
    """气象预测"""
    forecast_service = WeatherForecastService(db)
    result = forecast_service.forecast(
        station_id=forecast_request.station_id,
        forecast_days=forecast_request.forecast_days
    )
    return WeatherForecastResponse(**result)
