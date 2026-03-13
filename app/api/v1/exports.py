"""数据导出 API

此模块提供数据导出相关接口。
"""
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import io

from app.core.database import get_db
from app.core.export import export_service
from app.core.response import success_response
from fastapi.responses import StreamingResponse

from app.services.finance.stock_service import StockService
from app.services.weather.weather_service import WeatherService
from app.services.fortune.face_reading_service import FaceReadingService
from app.services.consumption.gdp_service import GDPService
from app.services.admin.user_service import UserService
from app.services.admin.operation_log_service import OperationLogService

router = APIRouter(prefix="/exports", tags=["数据导出"])


@router.get("/finance/stocks")
async def export_stocks(
    format: str = Query("csv", description="导出格式：csv 或 excel"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """导出股票数据"""
    service = StockService(db)
    stocks, total = await service.get_stocks(page=page, page_size=page_size)
    
    # 转换为字典列表
    data = []
    for stock in stocks:
        data.append({
            'ts_code': stock.ts_code,
            'trade_date': stock.trade_date,
            'open': float(stock.open) if stock.open else 0,
            'high': float(stock.high) if stock.high else 0,
            'low': float(stock.low) if stock.low else 0,
            'close': float(stock.close) if stock.close else 0,
            'volume': stock.volume,
            'amount': float(stock.amount) if stock.amount else 0
        })
    
    if format.lower() == 'csv':
        csv_bytes = export_service.export_to_csv_bytes(data)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=stocks.csv"}
        )
    else:
        excel_bytes = export_service.export_to_excel_bytes(data)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=stocks.xlsx"}
        )


@router.get("/weather/data")
async def export_weather_data(
    format: str = Query("csv", description="导出格式：csv 或 excel"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """导出气象数据"""
    service = WeatherService(db)
    weather_list, total = await service.get_weather_data(page=page, page_size=page_size)
    
    data = []
    for w in weather_list:
        data.append({
            'station_code': w.station_code,
            'observation_time': w.observation_time,
            'temperature': float(w.temperature) if w.temperature else 0,
            'humidity': float(w.humidity) if w.humidity else 0,
            'pressure': float(w.pressure) if w.pressure else 0,
            'wind_direction': w.wind_direction,
            'wind_speed': float(w.wind_speed) if w.wind_speed else 0,
            'weather': w.weather
        })
    
    if format.lower() == 'csv':
        csv_bytes = export_service.export_to_csv_bytes(data)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=weather_data.csv"}
        )
    else:
        excel_bytes = export_service.export_to_excel_bytes(data)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=weather_data.xlsx"}
        )


@router.get("/fortune/face-reading")
async def export_face_reading(
    format: str = Query("csv", description="导出格式：csv 或 excel"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """导出面相数据"""
    service = FaceReadingService(db)
    readings, total = await service.get_face_readings(page=page, page_size=page_size)
    
    data = []
    for r in readings:
        data.append({
            'name': r.name,
            'feature': r.feature,
            'feature_type': r.feature_type,
            'score': float(r.score) if r.score else 0,
            'fortune': r.fortune,
            'description': r.description
        })
    
    if format.lower() == 'csv':
        csv_bytes = export_service.export_to_csv_bytes(data)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=face_reading.csv"}
        )
    else:
        excel_bytes = export_service.export_to_excel_bytes(data)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=face_reading.xlsx"}
        )


@router.get("/consumption/gdp")
async def export_gdp_data(
    format: str = Query("csv", description="导出格式：csv 或 excel"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """导出 GDP 数据"""
    from app.services.consumption.gdp_service import GDPService
    service = GDPService(db)
    gdp_list, total = await service.get_gdp_data(page=page, page_size=page_size)
    
    data = []
    for g in gdp_list:
        data.append({
            'region': g.region,
            'year': g.year,
            'quarter': g.quarter,
            'gdp': float(g.gdp) if g.gdp else 0,
            'growth_rate': float(g.growth_rate) if g.growth_rate else 0
        })
    
    if format.lower() == 'csv':
        csv_bytes = export_service.export_to_csv_bytes(data)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=gdp_data.csv"}
        )
    else:
        excel_bytes = export_service.export_to_excel_bytes(data)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=gdp_data.xlsx"}
        )


@router.get("/admin/users")
async def export_users(
    format: str = Query("csv", description="导出格式：csv 或 excel"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """导出用户数据"""
    service = UserService(db)
    users, total = await service.get_users(page=page, page_size=page_size)
    
    data = []
    for u in users:
        data.append({
            'username': u.username,
            'email': u.email,
            'role': u.role.name if u.role else '',
            'status': 'active' if u.is_active else 'inactive',
            'created_at': u.created_at
        })
    
    if format.lower() == 'csv':
        csv_bytes = export_service.export_to_csv_bytes(data)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=users.csv"}
        )
    else:
        excel_bytes = export_service.export_to_excel_bytes(data)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=users.xlsx"}
        )


@router.get("/admin/logs")
async def export_operation_logs(
    format: str = Query("csv", description="导出格式：csv 或 excel"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """导出操作日志"""
    service = OperationLogService(db)
    logs, total = await service.get_logs(page=page, page_size=page_size)
    
    data = []
    for log in logs:
        data.append({
            'user_id': log.user_id,
            'username': log.username if log.username else '',
            'module': log.module,
            'action': log.action,
            'ip_address': log.ip_address,
            'created_at': log.created_at
        })
    
    if format.lower() == 'csv':
        csv_bytes = export_service.export_to_csv_bytes(data)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=operation_logs.csv"}
        )
    else:
        excel_bytes = export_service.export_to_excel_bytes(data)
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=operation_logs.xlsx"}
        )
