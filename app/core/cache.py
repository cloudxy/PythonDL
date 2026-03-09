"""缓存管理模块

此模块提供文件缓存功能，支持缓存的存储、获取、删除等操作。
"""
import json
import os
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Union
import threading
import logging

from app.core.config import config

logger = logging.getLogger(__name__)


class CacheManager:
    """文件缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: Optional[int] = None):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            ttl: 默认缓存过期时间（秒）
        """
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = ttl or config.CACHE_TTL
        self._lock = threading.Lock()
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            Path: 缓存文件路径
        """
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_meta_path(self, key: str) -> Path:
        """获取缓存元数据文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            Path: 元数据文件路径
        """
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值，如果不存在或已过期则返回None
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)
            
            if not cache_path.exists() or not meta_path.exists():
                return None
            
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                expire_time = meta.get('expire_time', 0)
                if expire_time > 0 and time.time() > expire_time:
                    self.delete(key)
                    return None
                
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
                return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒），0表示永不过期
            
        Returns:
            bool: 是否设置成功
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)
            
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                
                ttl = expire if expire is not None else self.default_ttl
                expire_time = time.time() + ttl if ttl > 0 else 0
                
                meta = {
                    'key': key,
                    'created_at': time.time(),
                    'expire_time': expire_time,
                    'ttl': ttl
                }
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                
                return True
            except Exception as e:
                logger.error(f"Error setting cache: {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)
            
            try:
                if cache_path.exists():
                    cache_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting cache: {str(e)}")
                return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """清空所有缓存
        
        Returns:
            bool: 是否清空成功
        """
        with self._lock:
            try:
                for file in self.cache_dir.glob("*.cache"):
                    file.unlink()
                for file in self.cache_dir.glob("*.meta"):
                    file.unlink()
                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                return False
    
    def get_ttl(self, key: str) -> Optional[int]:
        """获取缓存剩余过期时间
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[int]: 剩余过期时间（秒），如果不存在或永不过期则返回None
        """
        meta_path = self._get_meta_path(key)
        
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            expire_time = meta.get('expire_time', 0)
            if expire_time == 0:
                return None
            
            ttl = int(expire_time - time.time())
            return max(0, ttl)
        except Exception as e:
            logger.error(f"Error getting cache TTL: {str(e)}")
            return None
    
    def cleanup_expired(self) -> int:
        """清理过期缓存
        
        Returns:
            int: 清理的缓存数量
        """
        count = 0
        with self._lock:
            try:
                for meta_file in self.cache_dir.glob("*.meta"):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        
                        expire_time = meta.get('expire_time', 0)
                        if expire_time > 0 and time.time() > expire_time:
                            key = meta.get('key')
                            if key:
                                self.delete(key)
                                count += 1
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Error cleaning up expired cache: {str(e)}")
        
        return count
    
    def get_stats(self) -> dict:
        """获取缓存统计信息
        
        Returns:
            dict: 统计信息
        """
        total_size = 0
        total_count = 0
        expired_count = 0
        
        try:
            for file in self.cache_dir.glob("*.cache"):
                total_count += 1
                total_size += file.stat().st_size
            
            for meta_file in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    expire_time = meta.get('expire_time', 0)
                    if expire_time > 0 and time.time() > expire_time:
                        expired_count += 1
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
        
        return {
            'total_count': total_count,
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'expired_count': expired_count,
            'cache_dir': str(self.cache_dir)
        }


cache_manager = CacheManager()
