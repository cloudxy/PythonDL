"""金融分析API路由

此模块定义金融分析相关的API接口。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.schemas.finance import (
    StockBasicCreate,
    StockBasicUpdate,
    StockBasicResponse,
    StockDataCreate,
    StockDataResponse,
    StockPredictionRequest,
    StockPredictionResponse,
    StockRiskAssessmentResponse
)
from app.services.finance.stock_service import StockService
from app.services.finance.stock_prediction_service import StockPredictionService
from app.services.finance.stock_risk_service import StockRiskService

router = APIRouter()


@router.get("/stocks/basic", response_model=List[StockBasicResponse])
async def list_stock_basics(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    symbol: Optional[str] = None,
    industry: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取股票基础信息列表"""
    stock_service = StockService(db)
    stocks = stock_service.get_stock_basics(
        skip=skip,
        limit=limit,
        symbol=symbol,
        industry=industry
    )
    return [StockBasicResponse.from_orm(s) for s in stocks]


@router.post("/stocks/basic", response_model=StockBasicResponse)
async def create_stock_basic(
    stock_data: StockBasicCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("stock:create"))
):
    """创建股票基础信息"""
    stock_service = StockService(db)
    stock = stock_service.create_stock_basic(stock_data.dict())
    return StockBasicResponse.from_orm(stock)


@router.get("/stocks/basic/{stock_id}", response_model=StockBasicResponse)
async def get_stock_basic(
    stock_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取股票基础信息详情"""
    stock_service = StockService(db)
    stock = stock_service.get_stock_basic(stock_id)
    if not stock:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="股票不存在")
    return StockBasicResponse.from_orm(stock)


@router.put("/stocks/basic/{stock_id}", response_model=StockBasicResponse)
async def update_stock_basic(
    stock_id: int,
    stock_data: StockBasicUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("stock:update"))
):
    """更新股票基础信息"""
    stock_service = StockService(db)
    stock = stock_service.update_stock_basic(
        stock_id,
        stock_data.dict(exclude_unset=True)
    )
    return StockBasicResponse.from_orm(stock)


@router.delete("/stocks/basic/{stock_id}")
async def delete_stock_basic(
    stock_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("stock:delete"))
):
    """删除股票基础信息"""
    stock_service = StockService(db)
    stock_service.delete_stock_basic(stock_id)
    return {"message": "删除成功"}


@router.get("/stocks/data", response_model=List[StockDataResponse])
async def list_stock_data(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    ts_code: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取股票行情数据列表"""
    stock_service = StockService(db)
    data = stock_service.get_stock_data(
        skip=skip,
        limit=limit,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    return [StockDataResponse.from_orm(d) for d in data]


@router.post("/stocks/data", response_model=StockDataResponse)
async def create_stock_data(
    data: StockDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("stock:create"))
):
    """创建股票行情数据"""
    stock_service = StockService(db)
    stock_data = stock_service.create_stock_data(data.dict())
    return StockDataResponse.from_orm(stock_data)


@router.post("/stocks/predict", response_model=StockPredictionResponse)
async def predict_stock(
    prediction_request: StockPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("stock:predict"))
):
    """股票预测"""
    prediction_service = StockPredictionService(db)
    result = prediction_service.predict(
        ts_code=prediction_request.ts_code,
        prediction_days=prediction_request.prediction_days,
        model_type=prediction_request.model_type
    )
    return StockPredictionResponse(**result)


@router.get("/stocks/risk/{ts_code}", response_model=StockRiskAssessmentResponse)
async def assess_stock_risk(
    ts_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """股票风险评估"""
    risk_service = StockRiskService(db)
    result = risk_service.assess(ts_code)
    return StockRiskAssessmentResponse(**result)
