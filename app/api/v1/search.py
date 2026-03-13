"""
全局搜索 API
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.admin.user import User
from app.services.search_service import SearchService


router = APIRouter(prefix="/search", tags=["全局搜索"])


@router.get("", summary="全局搜索")
async def global_search(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    types: Optional[str] = Query(None, description="搜索类型，逗号分隔，如：users,stocks,weather"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    全局搜索接口
    
    支持搜索：
    - users: 用户
    - stocks: 股票数据
    - weather: 气象数据
    - face_analysis: 面相分析记录
    - operation_logs: 操作日志
    """
    service = SearchService(db)
    
    # 解析搜索类型
    search_types = None
    if types:
        search_types = [t.strip() for t in types.split(",")]
    
    try:
        results = await service.global_search(
            keyword=q,
            types=search_types,
            page=page,
            page_size=page_size
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败：{str(e)}")


@router.post("/advanced", summary="高级搜索")
async def advanced_search(
    model_type: str = Query(..., description="模型类型：stock, weather, user"),
    filters: dict = {},
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    高级搜索接口
    
    支持复杂的过滤条件搜索
    """
    service = SearchService(db)
    
    try:
        results = await service.advanced_search(
            filters=filters,
            model_type=model_type,
            page=page,
            page_size=page_size
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"高级搜索失败：{str(e)}")


@router.get("/suggestions", summary="搜索建议")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    limit: int = Query(10, ge=1, le=20, description="建议数量"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取搜索建议
    
    返回匹配的关键词建议
    """
    service = SearchService(db)
    
    suggestions = {
        "users": [],
        "stocks": [],
        "cities": []
    }
    
    # 获取用户建议
    try:
        from sqlalchemy import select
        from app.models.admin.user import User as UserModel
        
        query = select(UserModel.username).where(
            UserModel.username.ilike(f"%{q}%")
        ).limit(limit)
        
        result = await db.execute(query)
        suggestions["users"] = [row[0] for row in result.all()]
    except:
        pass
    
    # 获取股票建议
    try:
        from app.models.finance.stock import Stock
        
        query = select(Stock.ts_code).where(
            Stock.ts_code.ilike(f"%{q}%")
        ).limit(limit)
        
        result = await db.execute(query)
        suggestions["stocks"] = [row[0] for row in result.all()]
    except:
        pass
    
    # 获取城市建议
    try:
        from app.models.weather.weather import WeatherData
        
        query = select(WeatherData.city).where(
            WeatherData.city.ilike(f"%{q}%")
        ).distinct().limit(limit)
        
        result = await db.execute(query)
        suggestions["cities"] = [row[0] for row in result.all()]
    except:
        pass
    
    return suggestions
