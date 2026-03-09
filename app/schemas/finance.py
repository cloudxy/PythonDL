"""金融分析相关Schema

此模块定义金融分析相关的数据验证模型。
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal


class StockBasicCreate(BaseModel):
    """股票基础信息创建"""
    ts_code: str = Field(..., max_length=20)
    symbol: str = Field(..., max_length=10)
    name: str = Field(..., max_length=50)
    area: Optional[str] = Field(None, max_length=20)
    industry: Optional[str] = Field(None, max_length=50)
    market: Optional[str] = Field(None, max_length=20)
    exchange: Optional[str] = Field(None, max_length=20)
    list_status: str = Field(default="L", max_length=10)
    list_date: Optional[date] = None


class StockBasicUpdate(BaseModel):
    """股票基础信息更新"""
    name: Optional[str] = Field(None, max_length=50)
    area: Optional[str] = Field(None, max_length=20)
    industry: Optional[str] = Field(None, max_length=50)
    list_status: Optional[str] = Field(None, max_length=10)


class StockBasicResponse(BaseModel):
    """股票基础信息响应"""
    id: int
    ts_code: str
    symbol: str
    name: str
    area: Optional[str]
    industry: Optional[str]
    market: Optional[str]
    exchange: Optional[str]
    list_status: str
    list_date: Optional[date]
    created_at: datetime
    
    class Config:
        from_attributes = True


class StockDataCreate(BaseModel):
    """股票行情数据创建"""
    ts_code: str = Field(..., max_length=20)
    trade_date: date
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    pre_close: Optional[Decimal] = None
    change: Optional[Decimal] = None
    pct_chg: Optional[Decimal] = None
    vol: Optional[int] = None
    amount: Optional[Decimal] = None


class StockDataResponse(BaseModel):
    """股票行情数据响应"""
    id: int
    ts_code: str
    trade_date: date
    open: Optional[Decimal]
    high: Optional[Decimal]
    low: Optional[Decimal]
    close: Optional[Decimal]
    pre_close: Optional[Decimal]
    change: Optional[Decimal]
    pct_chg: Optional[Decimal]
    vol: Optional[int]
    amount: Optional[Decimal]
    ma5: Optional[Decimal]
    ma10: Optional[Decimal]
    ma20: Optional[Decimal]
    created_at: datetime
    
    class Config:
        from_attributes = True


class StockPredictionRequest(BaseModel):
    """股票预测请求"""
    ts_code: str = Field(..., max_length=20)
    prediction_days: int = Field(default=7, ge=1, le=30)
    model_type: Optional[str] = Field(default="lstm", max_length=50)


class StockPredictionResponse(BaseModel):
    """股票预测响应"""
    ts_code: str
    prediction_date: date
    predictions: List[dict]
    confidence: float
    model_type: str


class StockRiskAssessmentResponse(BaseModel):
    """股票风险评估响应"""
    ts_code: str
    assessment_date: date
    risk_score: float
    risk_level: str
    volatility: Optional[float]
    beta: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    var_95: Optional[float]
    var_99: Optional[float]
    liquidity_risk: Optional[float]
    market_risk: Optional[float]
    recommendations: Optional[str]
