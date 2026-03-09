"""消费分析相关Schema

此模块定义消费分析相关的数据验证模型。
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal


class GDPDataCreate(BaseModel):
    """GDP数据创建"""
    region_code: str = Field(..., max_length=20)
    region_name: str = Field(..., max_length=100)
    region_level: Optional[str] = Field(None, max_length=20)
    year: int
    quarter: Optional[int] = None
    gdp: Optional[int] = None
    gdp_growth: Optional[Decimal] = None
    primary_industry: Optional[int] = None
    secondary_industry: Optional[int] = None
    tertiary_industry: Optional[int] = None


class GDPDataResponse(BaseModel):
    """GDP数据响应"""
    id: int
    region_code: str
    region_name: str
    region_level: Optional[str]
    year: int
    quarter: Optional[int]
    gdp: Optional[int]
    gdp_growth: Optional[Decimal]
    primary_industry: Optional[int]
    secondary_industry: Optional[int]
    tertiary_industry: Optional[int]
    per_capita_gdp: Optional[Decimal]
    created_at: datetime
    
    class Config:
        from_attributes = True


class PopulationDataCreate(BaseModel):
    """人口数据创建"""
    region_code: str = Field(..., max_length=20)
    region_name: str = Field(..., max_length=100)
    region_level: Optional[str] = Field(None, max_length=20)
    year: int
    total_population: Optional[int] = None
    male_population: Optional[int] = None
    female_population: Optional[int] = None
    urban_population: Optional[int] = None
    rural_population: Optional[int] = None


class PopulationDataResponse(BaseModel):
    """人口数据响应"""
    id: int
    region_code: str
    region_name: str
    region_level: Optional[str]
    year: int
    total_population: Optional[int]
    male_population: Optional[int]
    female_population: Optional[int]
    urban_population: Optional[int]
    rural_population: Optional[int]
    urbanization_rate: Optional[Decimal]
    birth_rate: Optional[Decimal]
    death_rate: Optional[Decimal]
    created_at: datetime
    
    class Config:
        from_attributes = True


class EconomicIndicatorCreate(BaseModel):
    """经济指标数据创建"""
    region_code: str = Field(..., max_length=20)
    region_name: str = Field(..., max_length=100)
    region_level: Optional[str] = Field(None, max_length=20)
    year: int
    month: Optional[int] = None
    cpi: Optional[Decimal] = None
    ppi: Optional[Decimal] = None
    pmi: Optional[Decimal] = None
    retail_sales: Optional[int] = None
    fixed_asset_investment: Optional[int] = None


class EconomicIndicatorResponse(BaseModel):
    """经济指标数据响应"""
    id: int
    region_code: str
    region_name: str
    region_level: Optional[str]
    year: int
    month: Optional[int]
    cpi: Optional[Decimal]
    cpi_yoy: Optional[Decimal]
    ppi: Optional[Decimal]
    ppi_yoy: Optional[Decimal]
    pmi: Optional[Decimal]
    retail_sales: Optional[int]
    retail_sales_yoy: Optional[Decimal]
    created_at: datetime
    
    class Config:
        from_attributes = True


class CommunityDataCreate(BaseModel):
    """小区数据创建"""
    community_id: str = Field(..., max_length=50)
    community_name: str = Field(..., max_length=200)
    province: Optional[str] = Field(None, max_length=50)
    city: Optional[str] = Field(None, max_length=50)
    district: Optional[str] = Field(None, max_length=50)
    address: Optional[str] = Field(None, max_length=500)
    latitude: Optional[Decimal] = None
    longitude: Optional[Decimal] = None
    build_year: Optional[int] = None
    total_buildings: Optional[int] = None
    total_units: Optional[int] = None


class CommunityDataResponse(BaseModel):
    """小区数据响应"""
    id: int
    community_id: str
    community_name: str
    province: Optional[str]
    city: Optional[str]
    district: Optional[str]
    address: Optional[str]
    latitude: Optional[Decimal]
    longitude: Optional[Decimal]
    build_year: Optional[int]
    total_buildings: Optional[int]
    total_units: Optional[int]
    avg_price: Optional[Decimal]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConsumptionForecastRequest(BaseModel):
    """消费预测请求"""
    region_code: str = Field(..., max_length=20)
    forecast_months: int = Field(default=12, ge=1, le=36)


class ConsumptionForecastResponse(BaseModel):
    """消费预测响应"""
    region_code: str
    region_name: str
    forecast_date: date
    forecasts: List[dict]
    confidence: float
