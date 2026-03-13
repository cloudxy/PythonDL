"""数据文件管理模块

此模块提供数据文件的存储和管理功能。
"""
import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class DataFileManager:
    """数据文件管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.subdirs = {
            'finance': self.data_dir / 'finance',
            'weather': self.data_dir / 'weather',
            'fortune': self.data_dir / 'fortune',
            'consumption': self.data_dir / 'consumption',
            'exports': self.data_dir / 'exports',
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def save_json(
        self,
        data: Any,
        filename: str,
        category: str = 'exports'
    ) -> str:
        """保存 JSON 文件"""
        if category not in self.subdirs:
            category = 'exports'
        
        file_path = self.subdirs[category] / f"{filename}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def load_json(
        self,
        filename: str,
        category: str = 'exports'
    ) -> Optional[Any]:
        """加载 JSON 文件"""
        if category not in self.subdirs:
            category = 'exports'
        
        file_path = self.subdirs[category] / f"{filename}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        category: str = 'exports'
    ) -> str:
        """保存 CSV 文件"""
        if category not in self.subdirs:
            category = 'exports'
        
        file_path = self.subdirs[category] / f"{filename}.csv"
        
        if not data:
            return str(file_path)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return str(file_path)
    
    def load_csv(
        self,
        filename: str,
        category: str = 'exports'
    ) -> List[Dict[str, Any]]:
        """加载 CSV 文件"""
        if category not in self.subdirs:
            category = 'exports'
        
        file_path = self.subdirs[category] / f"{filename}.csv"
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def get_file_path(
        self,
        filename: str,
        category: str = 'exports'
    ) -> Optional[str]:
        """获取文件路径"""
        if category not in self.subdirs:
            category = 'exports'
        
        file_path = self.subdirs[category] / filename
        
        if not file_path.exists():
            return None
        
        return str(file_path)
    
    def delete_file(
        self,
        filename: str,
        category: str = 'exports'
    ) -> bool:
        """删除文件"""
        if category not in self.subdirs:
            category = 'exports'
        
        file_path = self.subdirs[category] / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except OSError:
                return False
        return False
    
    def list_files(
        self,
        category: str = 'exports',
        extension: Optional[str] = None
    ) -> List[str]:
        """列出文件"""
        if category not in self.subdirs:
            category = 'exports'
        
        files = []
        for file_path in self.subdirs[category].iterdir():
            if file_path.is_file():
                if extension is None or file_path.suffix == extension:
                    files.append(file_path.name)
        
        return sorted(files)


# 全局数据文件管理器
data_file_manager = DataFileManager()
