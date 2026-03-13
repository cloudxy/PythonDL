"""临时文件管理模块

此模块提供临时文件的创建、读取、删除功能。
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime


class TempFileManager:
    """临时文件管理器"""
    
    def __init__(self, temp_dir: str = "temps"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_temp_file(
        self,
        content: bytes,
        prefix: str = "temp_",
        suffix: str = "",
        expire_hours: int = 24
    ) -> str:
        """创建临时文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}{timestamp}{suffix}"
        file_path = self.temp_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    def get_temp_file(self, filename: str) -> Optional[bytes]:
        """获取临时文件内容"""
        file_path = self.temp_dir / filename
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def delete_temp_file(self, filename: str) -> bool:
        """删除临时文件"""
        file_path = self.temp_dir / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except OSError:
                return False
        return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """清理旧文件"""
        count = 0
        current_time = datetime.now()
        
        for file_path in self.temp_dir.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    try:
                        file_path.unlink()
                        count += 1
                    except OSError:
                        pass
        
        return count
    
    def clear_all(self) -> int:
        """清空所有临时文件"""
        count = 0
        for file_path in self.temp_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    count += 1
                except OSError:
                    pass
        return count


# 全局临时文件管理器
temp_file_manager = TempFileManager()
