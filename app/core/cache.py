"""文件缓存模块

此模块提供基于文件的缓存功能，支持 TTL 过期时间管理。
"""
import os
import json
import pickle
import time
import hashlib
from typing import Any, Optional, Union
from pathlib import Path


class FileCache:
    """基于文件的缓存系统"""
    
    def __init__(self, cache_dir: str = "runtimes/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.cache"
    
    def _is_expired(self, cache_path: Path) -> bool:
        """检查缓存是否过期"""
        if not cache_path.exists():
            return True
        
        # 读取缓存文件的元数据
        try:
            with open(cache_path, 'rb') as f:
                data = f.read()
                if len(data) < 100:  # 如果文件太小，可能是损坏的
                    return True
                
                # 解析缓存内容（前几位存储过期时间）
                content_str = data.decode('utf-8', errors='ignore')
                lines = content_str.split('\n', 2)
                if len(lines) >= 2:
                    try:
                        expire_time = float(lines[0])
                        return time.time() > expire_time
                    except ValueError:
                        return True
        except Exception:
            return True
        return False
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_path = self._get_cache_path(key)
        
        if self._is_expired(cache_path):
            # 删除过期的缓存文件
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = f.read()
                # 跳过第一行（过期时间）
                lines = data.decode('utf-8', errors='ignore').split('\n', 2)
                if len(lines) >= 2:
                    try:
                        # 从第二行开始是实际数据
                        serialized_data = '\n'.join(lines[1:])
                        return pickle.loads(serialized_data.encode('utf-8'))
                    except Exception:
                        return None
        except Exception:
            pass
        
        return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> None:
        """设置缓存值"""
        cache_path = self._get_cache_path(key)
        
        try:
            # 序列化值
            serialized_value = pickle.dumps(value).decode('utf-8')
            
            # 计算过期时间
            expire_time = time.time() + expire
            
            # 写入缓存文件
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(f"{expire_time}\n{serialized_value}")
        except Exception:
            pass
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                cache_path.unlink()
                return True
            except OSError:
                return False
        return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在且未过期"""
        cache_path = self._get_cache_path(key)
        return cache_path.exists() and not self._is_expired(cache_path)
    
    def clear_expired(self) -> int:
        """清理所有过期的缓存文件"""
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    count += 1
                except OSError:
                    pass
        return count
    
    def cleanup(self) -> int:
        """清理所有缓存文件"""
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass
        return count


# 全局缓存实例
cache_manager = FileCache()
