"""消费趋势分析 API

此模块提供消费趋势分析和图表数据接口。
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from app.core.database import get_db
from app.core.response import success_response
from app.services.consumption.gdp_service import GDPService
from app.services.consumption.economic_indicator_service import EconomicIndicatorService

router = APIRouter(prefix="/consumption/trends", tags=["消费趋势分析"])


@router.get("/gdp-trend")
async def get_gdp_trend(
    region: str = Query("national", description="地区"),
    start_year: int = Query(None, description="开始年份"),
    end_year: int = Query(None, description="结束年份"),
    db: AsyncSession = Depends(get_db)
):
    """获取 GDP 趋势数据"""
    service = GDPService(db)
    
    # 默认查询近 10 年
    if not start_year:
        start_year = datetime.now().year - 10
    if not end_year:
        end_year = datetime.now().year
    
    # 获取 GDP 数据
    gdp_data = await service.get_gdp_by_region(
        region=region,
        start_year=start_year,
        end_year=end_year
    )
    
    # 构建趋势数据
    trend_data = []
    for item in gdp_data:
        trend_data.append({
            "year": item.year,
            "gdp": float(item.gdp) if item.gdp else 0,
            "growth_rate": float(item.growth_rate) if item.growth_rate else 0
        })
    
    # 计算同比增长
    for i in range(1, len(trend_data)):
        prev_gdp = trend_data[i-1]["gdp"]
        curr_gdp = trend_data[i]["gdp"]
        if prev_gdp > 0:
            yoy_growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100
            trend_data[i]["yoy_growth"] = round(yoy_growth, 2)
        else:
            trend_data[i]["yoy_growth"] = 0
    
    return success_response({
        "region": region,
        "start_year": start_year,
        "end_year": end_year,
        "trend": trend_data
    })


@router.get("/indicator-trend")
async def get_indicator_trend(
    indicator_type: str = Query(..., description="指标类型"),
    region: str = Query("national", description="地区"),
    start_date: str = Query(None, description="开始日期"),
    end_date: str = Query(None, description="结束日期"),
    db: AsyncSession = Depends(get_db)
):
    """获取经济指标趋势数据"""
    service = EconomicIndicatorService(db)
    
    # 默认查询近 12 个月
    if not end_date:
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, "%Y-%m")
    
    if not start_date:
        start = end - relativedelta(months=11)
    else:
        start = datetime.strptime(start_date, "%Y-%m")
    
    # 获取指标数据
    indicators = await service.get_indicators_by_type(
        indicator_type=indicator_type,
        region=region,
        start_date=start,
        end_date=end
    )
    
    # 构建趋势数据
    trend_data = []
    for item in indicators:
        trend_data.append({
            "period": item.period,
            "value": float(item.value) if item.value else 0,
            "yoy_growth": float(item.yoy_growth) if item.yoy_growth else 0,
            "mom_growth": float(item.mom_growth) if item.mom_growth else 0
        })
    
    return success_response({
        "indicator_type": indicator_type,
        "region": region,
        "trend": trend_data
    })


@router.get("/comparison")
async def get_consumption_comparison(
    regions: str = Query(..., description="地区列表，逗号分隔"),
    year: int = Query(None, description="年份"),
    db: AsyncSession = Depends(get_db)
):
    """获取消费数据对比"""
    if not year:
        year = datetime.now().year - 1
    
    region_list = [r.strip() for r in regions.split(",")]
    
    comparison_data = []
    
    for region in region_list:
        # 获取 GDP 数据
        gdp_service = GDPService(db)
        gdp_data = await gdp_service.get_gdp_by_region(region=region, start_year=year, end_year=year)
        
        if gdp_data:
            comparison_data.append({
                "region": region,
                "gdp": float(gdp_data[0].gdp) if gdp_data[0].gdp else 0,
                "growth_rate": float(gdp_data[0].growth_rate) if gdp_data[0].growth_rate else 0
            })
    
    # 按 GDP 排序
    comparison_data.sort(key=lambda x: x["gdp"], reverse=True)
    
    # 添加排名
    for i, item in enumerate(comparison_data):
        item["rank"] = i + 1
    
    return success_response({
        "year": year,
        "regions": region_list,
        "comparison": comparison_data
    })


@router.get("/structure")
async def get_consumption_structure(
    year: int = Query(None, description="年份"),
    db: AsyncSession = Depends(get_db)
):
    """获取消费结构数据"""
    if not year:
        year = datetime.now().year - 1
    
    # 模拟消费结构数据
    structure_data = [
        {"category": "食品烟酒", "value": 30.0, "percentage": 30.0},
        {"category": "衣着", "value": 10.0, "percentage": 10.0},
        {"category": "居住", "value": 25.0, "percentage": 25.0},
        {"category": "生活用品", "value": 8.0, "percentage": 8.0},
        {"category": "交通通信", "value": 12.0, "percentage": 12.0},
        {"category": "教育文化", "value": 10.0, "percentage": 10.0},
        {"category": "医疗保健", "value": 5.0, "percentage": 5.0}
    ]
    
    return success_response({
        "year": year,
        "structure": structure_data
    })


@router.get("/forecast")
async def get_consumption_forecast(
    indicator: str = Query(..., description="预测指标"),
    periods: int = Query(12, description="预测期数"),
    db: AsyncSession = Depends(get_db)
):
    """获取消费预测数据"""
    # TODO: 实现预测算法
    # 这里返回模拟数据
    
    forecast_data = []
    base_value = 100
    
    for i in range(periods):
        forecast_date = datetime.now() + relativedelta(months=i+1)
        # 模拟增长
        value = base_value * (1 + 0.01 * (i + 1))
        
        forecast_data.append({
            "period": forecast_date.strftime("%Y-%m"),
            "forecast_value": round(value, 2),
            "lower_bound": round(value * 0.95, 2),
            "upper_bound": round(value * 1.05, 2)
        })
    
    return success_response({
        "indicator": indicator,
        "periods": periods,
        "forecast": forecast_data
    })
