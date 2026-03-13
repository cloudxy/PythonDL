"""
面相分析 API
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.admin.user import User
from app.schemas.fortune import FaceReadingCreate, FaceReadingResponse
from app.services.face.face_analysis_service import FaceAnalysisService
from app.services.face.image_upload_service import image_upload_service


router = APIRouter(prefix="/face-analysis", tags=["面相分析"])


@router.post("/upload", response_model=dict, summary="上传面相图片")
async def upload_face_image(
    file: UploadFile = File(..., description="面相图片文件"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """上传面相图片"""
    try:
        file_path, file_url = await image_upload_service.upload_face_image(
            file=file,
            user_id=current_user.id
        )
        
        return {
            "message": "上传成功",
            "file_path": file_path,
            "file_url": file_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败：{str(e)}")


@router.post("/analyze", response_model=FaceReadingResponse, summary="面相分析")
async def analyze_face(
    file: UploadFile = File(..., description="面相图片文件"),
    description: Optional[str] = Form(None, description="描述"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """上传面相图片并进行分析"""
    try:
        # 1. 上传图片
        file_path, file_url = await image_upload_service.upload_face_image(
            file=file,
            user_id=current_user.id
        )
        
        # 2. 进行面相分析
        service = FaceAnalysisService(db)
        analysis_result = await service.analyze_face_image(file_path)
        
        # 3. 保存分析记录
        analysis_data = FaceReadingCreate(
            name=f"Face Analysis {datetime.now().strftime('%Y%m%d%H%M%S')}",
            category="face_analysis",
            face_part=analysis_result.get("face_part", ""),
            feature_type=analysis_result.get("feature_type", ""),
            shape=analysis_result.get("shape", ""),
            meaning=analysis_result.get("meaning", ""),
            personality_traits=analysis_result.get("personality_traits", ""),
            fortune_indication=analysis_result.get("fortune_indication", ""),
            description=description,
            interpretation=json.dumps(analysis_result, ensure_ascii=False)
        )
        
        db_analysis = await service.create_analysis(analysis_data)
        
        return db_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败：{str(e)}")


@router.get("/{analysis_id}", response_model=FaceReadingResponse, summary="获取分析结果")
async def get_analysis_result(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取面相分析结果"""
    service = FaceAnalysisService(db)
    analysis = await service.get_analysis(analysis_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="分析记录不存在")
    
    # 检查权限
    if analysis.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="无权查看此分析记录")
    
    return analysis


@router.get("", response_model=list[FaceReadingResponse], summary="获取分析历史")
async def get_analysis_history(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户的面相分析历史记录"""
    service = FaceAnalysisService(db)
    analyses = await service.get_user_analyses(
        user_id=current_user.id,
        page=page,
        page_size=page_size
    )
    
    return analyses
