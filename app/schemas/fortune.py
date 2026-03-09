"""看相算命相关Schema

此模块定义看相算命相关的数据验证模型。
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal


class FengShuiCreate(BaseModel):
    """风水数据创建"""
    name: str = Field(..., max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    direction: Optional[str] = Field(None, max_length=20)
    element: Optional[str] = Field(None, max_length=20)
    trigram: Optional[str] = Field(None, max_length=20)
    lucky_numbers: Optional[str] = Field(None, max_length=50)
    lucky_colors: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    interpretation: Optional[str] = None


class FengShuiResponse(BaseModel):
    """风水数据响应"""
    id: int
    name: str
    category: Optional[str]
    direction: Optional[str]
    element: Optional[str]
    trigram: Optional[str]
    lucky_numbers: Optional[str]
    lucky_colors: Optional[str]
    description: Optional[str]
    interpretation: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class FaceReadingCreate(BaseModel):
    """面相数据创建"""
    name: str = Field(..., max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    face_part: Optional[str] = Field(None, max_length=50)
    feature_type: Optional[str] = Field(None, max_length=50)
    shape: Optional[str] = Field(None, max_length=50)
    meaning: Optional[str] = None
    personality_traits: Optional[str] = None
    fortune_indication: Optional[str] = None
    description: Optional[str] = None


class FaceReadingResponse(BaseModel):
    """面相数据响应"""
    id: int
    name: str
    category: Optional[str]
    face_part: Optional[str]
    feature_type: Optional[str]
    shape: Optional[str]
    meaning: Optional[str]
    personality_traits: Optional[str]
    fortune_indication: Optional[str]
    description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class BaziCreate(BaseModel):
    """八字数据创建"""
    name: str = Field(..., max_length=100)
    year_pillar: Optional[str] = Field(None, max_length=20)
    month_pillar: Optional[str] = Field(None, max_length=20)
    day_pillar: Optional[str] = Field(None, max_length=20)
    hour_pillar: Optional[str] = Field(None, max_length=20)
    day_master: Optional[str] = Field(None, max_length=20)
    life_analysis: Optional[str] = None
    personality: Optional[str] = None
    career: Optional[str] = None
    wealth: Optional[str] = None


class BaziResponse(BaseModel):
    """八字数据响应"""
    id: int
    name: str
    year_pillar: Optional[str]
    month_pillar: Optional[str]
    day_pillar: Optional[str]
    hour_pillar: Optional[str]
    day_master: Optional[str]
    life_analysis: Optional[str]
    personality: Optional[str]
    career: Optional[str]
    wealth: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ZhouYiResponse(BaseModel):
    """周易数据响应"""
    id: int
    hexagram_number: int
    hexagram_name: str
    upper_trigram: Optional[str]
    lower_trigram: Optional[str]
    judgment: Optional[str]
    image: Optional[str]
    meaning: Optional[str]
    interpretation: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConstellationResponse(BaseModel):
    """星座数据响应"""
    id: int
    name: str
    english_name: Optional[str]
    symbol: Optional[str]
    element: Optional[str]
    quality: Optional[str]
    ruling_planet: Optional[str]
    strengths: Optional[str]
    weaknesses: Optional[str]
    personality: Optional[str]
    love_style: Optional[str]
    career_style: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class FortuneTellingCreate(BaseModel):
    """运势数据创建"""
    name: str = Field(..., max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    fortune_type: Optional[str] = Field(None, max_length=50)
    period: Optional[str] = Field(None, max_length=50)
    target_date: Optional[date] = None
    overall_score: Optional[Decimal] = None
    overall_fortune: Optional[str] = None
    career_score: Optional[Decimal] = None
    career_fortune: Optional[str] = None
    wealth_score: Optional[Decimal] = None
    wealth_fortune: Optional[str] = None


class FortuneTellingResponse(BaseModel):
    """运势数据响应"""
    id: int
    name: str
    category: Optional[str]
    fortune_type: Optional[str]
    period: Optional[str]
    target_date: Optional[date]
    overall_score: Optional[Decimal]
    overall_fortune: Optional[str]
    career_score: Optional[Decimal]
    career_fortune: Optional[str]
    wealth_score: Optional[Decimal]
    wealth_fortune: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class FortuneAnalysisRequest(BaseModel):
    """看相算命分析请求"""
    analysis_type: str = Field(..., max_length=50)
    params: Dict[str, Any] = Field(default_factory=dict)


class FortuneAnalysisResponse(BaseModel):
    """看相算命分析响应"""
    analysis_type: str
    result: Dict[str, Any]
    interpretation: str
    recommendations: List[str]
