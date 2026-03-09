"""看相算命API路由

此模块定义看相算命相关的API接口。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.auth import get_current_user, require_permission
from app.models.admin.user import User
from app.schemas.fortune import (
    FengShuiCreate,
    FengShuiResponse,
    FaceReadingCreate,
    FaceReadingResponse,
    BaziCreate,
    BaziResponse,
    ZhouYiResponse,
    ConstellationResponse,
    FortuneTellingCreate,
    FortuneTellingResponse,
    FortuneAnalysisRequest,
    FortuneAnalysisResponse
)
from app.services.fortune.feng_shui_service import FengShuiService
from app.services.fortune.face_reading_service import FaceReadingService
from app.services.fortune.bazi_service import BaziService
from app.services.fortune.zhou_yi_service import ZhouYiService
from app.services.fortune.constellation_service import ConstellationService
from app.services.fortune.fortune_telling_service import FortuneTellingService
from app.services.fortune.fortune_analysis_service import FortuneAnalysisService

router = APIRouter()


@router.get("/feng-shui", response_model=List[FengShuiResponse])
async def list_feng_shui(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取风水数据列表"""
    service = FengShuiService(db)
    data = service.get_feng_shui_list(skip=skip, limit=limit, category=category)
    return [FengShuiResponse.from_orm(d) for d in data]


@router.post("/feng-shui", response_model=FengShuiResponse)
async def create_feng_shui(
    data: FengShuiCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:create"))
):
    """创建风水数据"""
    service = FengShuiService(db)
    item = service.create_feng_shui(data.dict())
    return FengShuiResponse.from_orm(item)


@router.put("/feng-shui/{item_id}", response_model=FengShuiResponse)
async def update_feng_shui(
    item_id: int,
    data: FengShuiCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:update"))
):
    """更新风水数据"""
    service = FengShuiService(db)
    item = service.update_feng_shui(item_id, data.dict(exclude_unset=True))
    return FengShuiResponse.from_orm(item)


@router.delete("/feng-shui/{item_id}")
async def delete_feng_shui(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:delete"))
):
    """删除风水数据"""
    service = FengShuiService(db)
    service.delete_feng_shui(item_id)
    return {"message": "删除成功"}


@router.get("/face-reading", response_model=List[FaceReadingResponse])
async def list_face_reading(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    face_part: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取面相数据列表"""
    service = FaceReadingService(db)
    data = service.get_face_reading_list(skip=skip, limit=limit, face_part=face_part)
    return [FaceReadingResponse.from_orm(d) for d in data]


@router.post("/face-reading", response_model=FaceReadingResponse)
async def create_face_reading(
    data: FaceReadingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:create"))
):
    """创建面相数据"""
    service = FaceReadingService(db)
    item = service.create_face_reading(data.dict())
    return FaceReadingResponse.from_orm(item)


@router.put("/face-reading/{item_id}", response_model=FaceReadingResponse)
async def update_face_reading(
    item_id: int,
    data: FaceReadingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:update"))
):
    """更新面相数据"""
    service = FaceReadingService(db)
    item = service.update_face_reading(item_id, data.dict(exclude_unset=True))
    return FaceReadingResponse.from_orm(item)


@router.delete("/face-reading/{item_id}")
async def delete_face_reading(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:delete"))
):
    """删除面相数据"""
    service = FaceReadingService(db)
    service.delete_face_reading(item_id)
    return {"message": "删除成功"}


@router.get("/bazi", response_model=List[BaziResponse])
async def list_bazi(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取八字数据列表"""
    service = BaziService(db)
    data = service.get_bazi_list(skip=skip, limit=limit)
    return [BaziResponse.from_orm(d) for d in data]


@router.post("/bazi", response_model=BaziResponse)
async def create_bazi(
    data: BaziCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:create"))
):
    """创建八字数据"""
    service = BaziService(db)
    item = service.create_bazi(data.dict())
    return BaziResponse.from_orm(item)


@router.get("/zhou-yi", response_model=List[ZhouYiResponse])
async def list_zhou_yi(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取周易数据列表"""
    service = ZhouYiService(db)
    data = service.get_zhou_yi_list(skip=skip, limit=limit)
    return [ZhouYiResponse.from_orm(d) for d in data]


@router.get("/constellation", response_model=List[ConstellationResponse])
async def list_constellation(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取星座数据列表"""
    service = ConstellationService(db)
    data = service.get_constellation_list(skip=skip, limit=limit)
    return [ConstellationResponse.from_orm(d) for d in data]


@router.get("/fortune-telling", response_model=List[FortuneTellingResponse])
async def list_fortune_telling(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取运势数据列表"""
    service = FortuneTellingService(db)
    data = service.get_fortune_telling_list(skip=skip, limit=limit, category=category)
    return [FortuneTellingResponse.from_orm(d) for d in data]


@router.post("/fortune-telling", response_model=FortuneTellingResponse)
async def create_fortune_telling(
    data: FortuneTellingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("fortune:create"))
):
    """创建运势数据"""
    service = FortuneTellingService(db)
    item = service.create_fortune_telling(data.dict())
    return FortuneTellingResponse.from_orm(item)


@router.post("/analyze", response_model=FortuneAnalysisResponse)
async def analyze_fortune(
    request: FortuneAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """看相算命分析"""
    service = FortuneAnalysisService(db)
    result = service.analyze(
        analysis_type=request.analysis_type,
        params=request.params
    )
    return FortuneAnalysisResponse(**result)
