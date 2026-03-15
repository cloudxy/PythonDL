"""请求去重模块

此模块提供请求指纹生成和去重功能，避免重复爬取。
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, Any
from collections import OrderedDict

from redis.asyncio import Redis

from app.core.logger import get_logger
from app.core.redis_client import get_redis_client

logger = get_logger("request_dedup")


class RequestFingerprintGenerator:
    """请求指纹生成器"""
    
    @staticmethod
    def generate(url: str, method: str = "GET", params: Optional[Dict] = None, 
                 data: Optional[Dict] = None, headers: Optional[Dict] = None) -> str:
        """生成请求指纹"""
        # 构建指纹数据
        fingerprint_data = {
            "url": url,
            "method": method.upper(),
            "params": params or {},
            "data": data or {},
        }
        
        # 序列化并计算 MD5
        data_str = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.md5(data_str.encode()).hexdigest()
        
        logger.debug(f"生成请求指纹：{fingerprint[:16]}...")
        return fingerprint
    
    @staticmethod
    def generate_simple(url: str) -> str:
        """生成简化指纹（仅 URL）"""
        return hashlib.md5(url.encode()).hexdigest()
    
    @staticmethod
    def generate_with_timestamp(url: str, timestamp: Optional[int] = None) -> str:
        """生成带时间戳的指纹（用于定期重爬）"""
        if not timestamp:
            timestamp = int(time.time() / 3600)  # 按小时分组
        
        data_str = f"{url}:{timestamp}"
        return hashlib.md5(data_str.encode()).hexdigest()


class RequestDeduplicator:
    """请求去重器"""
    
    def __init__(
        self,
        use_redis: bool = True,
        max_memory_size: int = 100000,
        ttl: int = 86400 * 7  # 7 天
    ):
        self.use_redis = use_redis
        self.max_memory_size = max_memory_size
        self.ttl = ttl
        
        # 内存去重（LRU）
        self.memory_set: OrderedDict[str, float] = OrderedDict()
        
        # Redis 客户端
        self.redis: Optional[Redis] = None
        self.redis_key_prefix = "crawler:duplicate:"
    
    async def initialize(self):
        """初始化"""
        if self.use_redis:
            try:
                self.redis = get_redis_client()
                await self.redis.ping()
                logger.info("请求去重器 Redis 初始化成功")
            except Exception as e:
                logger.warning(f"请求去重器 Redis 初始化失败：{e}")
                self.use_redis = False
    
    async def is_duplicate(self, fingerprint: str) -> bool:
        """检查是否重复"""
        # 检查内存
        if fingerprint in self.memory_set:
            logger.debug(f"内存去重：{fingerprint[:16]}...")
            return True
        
        # 检查 Redis
        if self.use_redis and self.redis:
            try:
                key = f"{self.redis_key_prefix}{fingerprint}"
                exists = await self.redis.exists(key)
                if exists:
                    logger.debug(f"Redis 去重：{fingerprint[:16]}...")
                    return True
            except Exception as e:
                logger.warning(f"Redis 检查去重失败：{e}")
        
        return False
    
    async def add(self, fingerprint: str) -> bool:
        """添加到去重集合"""
        # 添加到内存
        if len(self.memory_set) >= self.max_memory_size:
            # 移除最旧的
            self.memory_set.popitem(last=False)
        
        self.memory_set[fingerprint] = time.time()
        
        # 添加到 Redis
        if self.use_redis and self.redis:
            try:
                key = f"{self.redis_key_prefix}{fingerprint}"
                await self.redis.setex(key, self.ttl, "1")
            except Exception as e:
                logger.warning(f"Redis 添加去重失败：{e}")
                return False
        
        return True
    
    async def check_and_add(self, fingerprint: str) -> bool:
        """检查并添加（原子操作）"""
        is_dup = await self.is_duplicate(fingerprint)
        
        if not is_dup:
            await self.add(fingerprint)
            return False  # 不是重复
        
        return True  # 是重复
    
    async def remove(self, fingerprint: str):
        """从去重集合中移除"""
        # 从内存移除
        if fingerprint in self.memory_set:
            del self.memory_set[fingerprint]
        
        # 从 Redis 移除
        if self.use_redis and self.redis:
            try:
                key = f"{self.redis_key_prefix}{fingerprint}"
                await self.redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis 删除去重失败：{e}")
    
    async def clear(self):
        """清除去重集合"""
        self.memory_set.clear()
        
        if self.use_redis and self.redis:
            try:
                # 删除所有相关 key
                pattern = f"{self.redis_key_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self.redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis 清除去重失败：{e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "memory_size": len(self.memory_set),
            "max_memory_size": self.max_memory_size,
            "use_redis": self.use_redis,
            "ttl": self.ttl
        }


class URLFilter:
    """URL 过滤器"""
    
    def __init__(self):
        # 黑名单域名
        self.blacklist_domains: Set[str] = set()
        # 白名单域名
        self.whitelist_domains: Set[str] = set()
        # 黑名单 URL 模式
        self.blacklist_patterns: Set[str] = set()
        # 允许的 URL 最大长度
        self.max_url_length = 2048
    
    def add_blacklist_domain(self, domain: str):
        """添加黑名单域名"""
        self.blacklist_domains.add(domain)
        logger.info(f"添加黑名单域名：{domain}")
    
    def add_whitelist_domain(self, domain: str):
        """添加白名单域名"""
        self.whitelist_domains.add(domain)
        logger.info(f"添加白名单域名：{domain}")
    
    def add_blacklist_pattern(self, pattern: str):
        """添加黑名单 URL 模式"""
        self.blacklist_patterns.add(pattern)
        logger.info(f"添加黑名单模式：{pattern}")
    
    def is_allowed(self, url: str) -> bool:
        """检查 URL 是否允许"""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            
            # 检查 URL 长度
            if len(url) > self.max_url_length:
                logger.debug(f"URL 过长：{url[:100]}...")
                return False
            
            # 检查协议
            if parsed.scheme not in ['http', 'https']:
                logger.debug(f"不支持的协议：{parsed.scheme}")
                return False
            
            # 检查域名
            domain = parsed.netloc
            
            # 如果有白名单，检查是否在白名单中
            if self.whitelist_domains:
                if domain not in self.whitelist_domains:
                    logger.debug(f"域名不在白名单：{domain}")
                    return False
            
            # 检查黑名单域名
            if domain in self.blacklist_domains:
                logger.debug(f"域名在黑名单：{domain}")
                return False
            
            # 检查黑名单模式
            for pattern in self.blacklist_patterns:
                if pattern in url:
                    logger.debug(f"URL 匹配黑名单模式：{pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"检查 URL 失败：{e}")
            return False


# 全局去重器实例
_deduplicator: Optional[RequestDeduplicator] = None
_url_filter: Optional[URLFilter] = None


def get_deduplicator() -> RequestDeduplicator:
    """获取去重器实例"""
    global _deduplicator
    
    if _deduplicator is None:
        _deduplicator = RequestDeduplicator()
    
    return _deduplicator


def get_url_filter() -> URLFilter:
    """获取 URL 过滤器实例"""
    global _url_filter
    
    if _url_filter is None:
        _url_filter = URLFilter()
    
    return _url_filter


async def init_deduplicator():
    """初始化去重器"""
    deduplicator = get_deduplicator()
    await deduplicator.initialize()
    logger.info("请求去重器初始化完成")
