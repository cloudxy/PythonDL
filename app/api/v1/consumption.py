"""消费分析API路由

此模块定义消费分析相关的API接口。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.schemas.consumption import (
    GDPDataCreate,
    GDPDataResponse,
    PopulationDataCreate,
    PopulationDataResponse,
    EconomicIndicatorCreate,
    EconomicIndicatorResponse,
    CommunityDataCreate,
    CommunityDataResponse,
    ConsumptionForecastRequest,
    ConsumptionForecastResponse
)
from app.services.consumption.gdp_service import GDPService
from app.services.consumption.population_service import PopulationService
from app.services.consumption.economic_indicator_service import EconomicIndicatorService
from app.services.consumption.community_service import CommunityService
from app.services.consumption.consumption_forecast_service import ConsumptionForecastService

router = APIRouter()


@router.get("/gdp", response_model=List[GDPDataResponse])
async def list_gdp(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    region_code: Optional[str] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取GDP数据列表"""
    service = GDPService(db)
    data = service.get_gdp_data(
        skip=skip,
        limit=limit,
        region_code=region_code,
        year=year
    )
    return [GDPDataResponse.from_orm(d) for d in data]


@router.post("/gdp", response_model=GDPDataResponse)
async def create_gdp(
    data: GDPDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:create"))
):
    """创建GDP数据"""
    service = GDPService(db)
    item = service.create_gdp_data(data.dict())
    return GDPDataResponse.from_orm(item)


@router.put("/gdp/{item_id}", response_model=GDPDataResponse)
async def update_gdp(
    item_id: int,
    data: GDPDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:update"))
):
    """更新GDP数据"""
    service = GDPService(db)
    item = service.update_gdp_data(item_id, data.dict(exclude_unset=True))
    return GDPDataResponse.from_orm(item)


@router.delete("/gdp/{item_id}")
async def delete_gdp(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:delete"))
):
    """删除GDP数据"""
    service = GDPService(db)
    service.delete_gdp_data(item_id)
    return {"message": "删除成功"}


@router.get("/population", response_model=List[PopulationDataResponse])
async def list_population(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    region_code: Optional[str] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取人口数据列表"""
    service = PopulationService(db)
    data = service.get_population_data(
        skip=skip,
        limit=limit,
        region_code=region_code,
        year=year
    )
    return [PopulationDataResponse.from_orm(d) for d in data]


@router.post("/population", response_model=PopulationDataResponse)
async def create_population(
    data: PopulationDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:create"))
):
    """创建人口数据"""
    service = PopulationService(db)
    item = service.create_population_data(data.dict())
    return PopulationDataResponse.from_orm(item)


@router.put("/population/{item_id}", response_model=PopulationDataResponse)
async def update_population(
    item_id: int,
    data: PopulationDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:update"))
):
    """更新人口数据"""
    service = PopulationService(db)
    item = service.update_population_data(item_id, data.dict(exclude_unset=True))
    return PopulationDataResponse.from_orm(item)


@router.delete("/population/{item_id}")
async def delete_population(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:delete"))
):
    """删除人口数据"""
    service = PopulationService(db)
    service.delete_population_data(item_id)
    return {"message": "删除成功"}


@router.get("/economic-indicators", response_model=List[EconomicIndicatorResponse])
async def list_economic_indicators(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    region_code: Optional[str] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取经济指标数据列表"""
    service = EconomicIndicatorService(db)
    data = service.get_economic_indicators(
        skip=skip,
        limit=limit,
        region_code=region_code,
        year=year
    )
    return [EconomicIndicatorResponse.from_orm(d) for d in data]


@router.post("/economic-indicators", response_model=EconomicIndicatorResponse)
async def create_economic_indicator(
    data: EconomicIndicatorCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:create"))
):
    """创建经济指标数据"""
    service = EconomicIndicatorService(db)
    item = service.create_economic_indicator(data.dict())
    return EconomicIndicatorResponse.from_orm(item)


@router.get("/community", response_model=List[CommunityDataResponse])
async def list_community(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    city: Optional[str] = None,
    district: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取小区数据列表"""
    service = CommunityService(db)
    data = service.get_community_data(
        skip=skip,
        limit=limit,
        city=city,
        district=district
    )
    return [CommunityDataResponse.from_orm(d) for d in data]


@router.post("/community", response_model=CommunityDataResponse)
async def create_community(
    data: CommunityDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:create"))
):
    """创建小区数据"""
    service = CommunityService(db)
    item = service.create_community_data(data.dict())
    return CommunityDataResponse.from_orm(item)


@router.put("/community/{item_id}", response_model=CommunityDataResponse)
async def update_community(
    item_id: int,
    data: CommunityDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:update"))
):
    """更新小区数据"""
    service = CommunityService(db)
    item = service.update_community_data(item_id, data.dict(exclude_unset=True))
    return CommunityDataResponse.from_orm(item)


@router.delete("/community/{item_id}")
async def delete_community(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:delete"))
):
    """删除小区数据"""
    service = CommunityService(db)
    service.delete_community_data(item_id)
    return {"message": "删除成功"}


@router.post("/forecast", response_model=ConsumptionForecastResponse)
async def forecast_consumption(
    request: ConsumptionForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("consumption:forecast"))
):
    """宏观消费预测"""
    service = ConsumptionForecastService(db)
    result = service.forecast(
        region_code=request.region_code,
        forecast_months=request.forecast_months
    )
    return ConsumptionForecastResponse(**result)
