"""
图片上传服务
"""
import os
import uuid
import aiofiles
from typing import Optional, Tuple
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.core.config import settings


class ImageUploadService:
    """图片上传服务类"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent.parent / "files" / "face_images"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的文件类型
        self.allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        # 最大文件大小 (10MB)
        self.max_file_size = 10 * 1024 * 1024
    
    async def upload_face_image(
        self,
        file: UploadFile,
        user_id: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        上传面相图片
        
        Args:
            file: 上传的文件
            user_id: 用户 ID
        
        Returns:
            (file_path, file_url) 元组
        """
        # 验证文件类型
        if not self._validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型，仅支持：{', '.join(self.allowed_extensions)}"
            )
        
        # 生成唯一文件名
        file_extension = self._get_file_extension(file.filename)
        filename = f"{uuid.uuid4().hex}{file_extension}"
        
        # 创建用户子目录
        if user_id:
            user_dir = self.base_dir / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            file_path = user_dir / filename
            relative_path = f"files/face_images/{user_id}/{filename}"
        else:
            file_path = self.base_dir / filename
            relative_path = f"files/face_images/{filename}"
        
        # 保存文件
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                
                # 检查文件大小
                if len(content) > self.max_file_size:
                    raise HTTPException(
                        status_code=400,
                        detail="文件大小超过限制 (10MB)"
                    )
                
                await out_file.write(content)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"文件保存失败：{str(e)}"
            )
        
        # 生成 URL
        file_url = f"/{relative_path}"
        
        return str(file_path), file_url
    
    def _validate_file_type(self, filename: str) -> bool:
        """验证文件类型"""
        if not filename:
            return False
        
        extension = self._get_file_extension(filename)
        return extension.lower() in self.allowed_extensions
    
    def _get_file_extension(self, filename: str) -> str:
        """获取文件扩展名"""
        if not filename:
            return ""
        
        parts = filename.rsplit('.', 1)
        if len(parts) < 2:
            return ""
        
        return "." + parts[1].lower()
    
    async def delete_image(self, file_path: str) -> bool:
        """删除图片"""
        try:
            path = Path(file_path)
            if path.exists():
                os.remove(path)
                return True
            return False
        except Exception:
            return False
    
    def get_image_url(self, relative_path: str) -> str:
        """获取图片 URL"""
        return f"/{relative_path}"


# 全局实例
image_upload_service = ImageUploadService()
