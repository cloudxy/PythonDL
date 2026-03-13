"""导出文件管理模块

此模块管理程序运行过程中导出的文件、图片、视频。
"""
import os
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class ExportFileManager:
    """导出文件管理器"""
    
    def __init__(self, files_dir: str = "files"):
        self.files_dir = Path(files_dir)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.exports_dir = self.files_dir / 'exports'
        self.images_dir = self.files_dir / 'images'
        self.videos_dir = self.files_dir / 'videos'
        
        for dir_path in [self.exports_dir, self.images_dir, self.videos_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_export(
        self,
        content: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """保存导出文件"""
        if subfolder:
            export_dir = self.exports_dir / subfolder
            export_dir.mkdir(parents=True, exist_ok=True)
        else:
            export_dir = self.exports_dir
        
        file_path = export_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    def save_image(
        self,
        image_data: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """保存图片文件"""
        if subfolder:
            image_dir = self.images_dir / subfolder
            image_dir.mkdir(parents=True, exist_ok=True)
        else:
            image_dir = self.images_dir
        
        file_path = image_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return str(file_path)
    
    def save_video(
        self,
        video_data: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """保存视频文件"""
        if subfolder:
            video_dir = self.videos_dir / subfolder
            video_dir.mkdir(parents=True, exist_ok=True)
        else:
            video_dir = self.videos_dir
        
        file_path = video_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(video_data)
        
        return str(file_path)
    
    def get_export(self, filename: str, subfolder: Optional[str] = None) -> Optional[bytes]:
        """获取导出文件"""
        if subfolder:
            file_path = self.exports_dir / subfolder / filename
        else:
            file_path = self.exports_dir / filename
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def get_image(self, filename: str, subfolder: Optional[str] = None) -> Optional[bytes]:
        """获取图片文件"""
        if subfolder:
            file_path = self.images_dir / subfolder / filename
        else:
            file_path = self.images_dir / filename
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def get_video(self, filename: str, subfolder: Optional[str] = None) -> Optional[bytes]:
        """获取视频文件"""
        if subfolder:
            file_path = self.videos_dir / subfolder / filename
        else:
            file_path = self.videos_dir / filename
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def delete_export(self, filename: str, subfolder: Optional[str] = None) -> bool:
        """删除导出文件"""
        if subfolder:
            file_path = self.exports_dir / subfolder / filename
        else:
            file_path = self.exports_dir / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except OSError:
                return False
        return False
    
    def list_exports(
        self,
        folder: str = 'exports',
        extension: Optional[str] = None
    ) -> List[str]:
        """列出文件"""
        if folder == 'exports':
            base_dir = self.exports_dir
        elif folder == 'images':
            base_dir = self.images_dir
        elif folder == 'videos':
            base_dir = self.videos_dir
        else:
            return []
        
        files = []
        for file_path in base_dir.rglob('*'):
            if file_path.is_file():
                if extension is None or file_path.suffix == extension:
                    files.append(str(file_path.relative_to(base_dir)))
        
        return sorted(files)
    
    def get_file_url(self, filename: str, folder: str = 'exports') -> str:
        """获取文件 URL"""
        return f"/files/{folder}/{filename}"


# 全局导出文件管理器
export_file_manager = ExportFileManager()
