"""统一异步爬虫基类

此模块提供所有爬虫的统一基类，包含通用的 HTTP 请求、缓存、频率控制、防爬等功能。
"""
import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from redis.asyncio import Redis

from app.core.logger import get_logger
from app.core.redis_client import get_redis_client
from app.services.crawler.anti_crawling import get_anti_crawling_manager, AntiCrawlingManager
from app.services.crawler.request_dedup import get_deduplicator, RequestDeduplicator, get_url_filter, URLFilter
from app.services.crawler.captcha_solver import get_captcha_handler, CaptchaHandler

logger = get_logger("base_crawler")

T = TypeVar('T')


@dataclass
class CrawlerStatus:
    """爬虫状态数据类"""
    is_running: bool = False
    last_run: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_records: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_records": self.total_records,
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class RequestConfig:
    """请求配置"""
    timeout: int = 30
    retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    max_concurrent: int = 10
    headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    })


class CircuitBreakerError(Exception):
    """熔断器异常"""
    pass


class CrawlerCircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, fail_max: int = 5, reset_timeout: int = 60):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = 'closed'  # closed, open, half-open
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = 'closed'
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.fail_max:
            self.state = 'open'
            logger.warning(f"熔断器已打开，失败次数：{self.failure_count}")
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == 'closed':
            return True
        
        if self.state == 'open':
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.reset_timeout:
                logger.info("熔断器进入半开状态")
                self.state = 'half-open'
                return True
            return False
        
        # half-open 状态，允许一次尝试
        return True
    
    async def execute(self, func: Callable, *args, **kwargs):
        """带熔断器的执行"""
        if not self.can_execute():
            raise CircuitBreakerError("熔断器已打开，请求被拒绝")
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class AsyncBaseCrawler(ABC, Generic[T]):
    """异步爬虫基类
    
    提供通用的 HTTP 请求、缓存、频率控制、熔断器等功能
    """
    
    def __init__(
        self,
        db: Optional[Any] = None,
        config: Optional[RequestConfig] = None,
        crawler_type: str = "base"
    ):
        self.db = db
        self.config = config or RequestConfig()
        self.crawler_type = crawler_type
        
        # 会话和连接池
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector = TCPConnector(
            limit=self.config.max_concurrent,
            limit_per_host=self.config.max_concurrent // 2,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        # Redis 缓存
        self.redis: Optional[Redis] = None
        self.use_cache = True
        
        # 熔断器
        self.circuit_breaker = CrawlerCircuitBreaker()
        
        # 频率控制
        self.last_request_time: float = 0
        self.request_count: int = 0
        
        # 状态
        self.status = CrawlerStatus()
        
        # 信号量控制并发
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # 防爬组件
        self.anti_crawling: Optional[AntiCrawlingManager] = None
        self.deduplicator: Optional[RequestDeduplicator] = None
        self.url_filter: Optional[URLFilter] = None
        self.captcha_handler: Optional[CaptchaHandler] = None
        self.use_anti_crawling = True
        self.use_deduplication = True
    
    async def initialize(self):
        """初始化爬虫"""
        try:
            # 创建 HTTP 会话
            timeout = ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers=self.config.headers
            )
            
            # 初始化 Redis
            try:
                self.redis = get_redis_client()
                await self.redis.ping()
                logger.info(f"{self.crawler_type} 爬虫 Redis 连接成功")
            except Exception as e:
                logger.warning(f"{self.crawler_type} 爬虫 Redis 连接失败：{e}")
                self.use_cache = False
            
            logger.info(f"{self.crawler_type} 爬虫初始化完成")
            
        except Exception as e:
            logger.error(f"{self.crawler_type} 爬虫初始化失败：{e}")
            raise
    
    async def close(self):
        """关闭爬虫，释放资源"""
        if self.session:
            await self.session.close()
            logger.info(f"{self.crawler_type} 爬虫会话已关闭")
        
        if self.redis:
            await self.redis.close()
            logger.info(f"{self.crawler_type} 爬虫 Redis 连接已关闭")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _rate_limit(self):
        """请求频率控制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - elapsed
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        use_circuit_breaker: bool = True
    ) -> Optional[aiohttp.ClientResponse]:
        """发送 HTTP 请求（带重试和熔断）"""
        
        async def _do_request():
            self.status.total_requests += 1
            
            await self._rate_limit()
            
            request_headers = self.config.headers.copy()
            if headers:
                request_headers.update(headers)
            
            async with self.session.request(
                method,
                url,
                params=params,
                data=data,
                headers=request_headers
            ) as response:
                response.raise_for_status()
                return response
        
        for attempt in range(self.config.retries):
            try:
                if use_circuit_breaker:
                    response = await self.circuit_breaker.execute(_do_request)
                else:
                    response = await _do_request()
                
                self.status.successful_requests += 1
                return response
                
            except CircuitBreakerError:
                logger.error(f"熔断器阻止请求：{url}")
                self.status.failed_requests += 1
                self.status.error_count += 1
                return None
                
            except Exception as e:
                logger.warning(f"请求失败 (第{attempt+1}次): {url}, 错误：{e}")
                
                if attempt == self.config.retries - 1:
                    logger.error(f"请求最终失败：{url}")
                    self.status.failed_requests += 1
                    self.status.error_count += 1
                    return None
                
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return None
    
    async def _cache_get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.use_cache or not self.redis:
            return None
        
        try:
            cached = await self.redis.get(key)
            if cached:
                self.status.cache_hits += 1
                logger.debug(f"缓存命中：{key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"读取缓存失败：{key}, 错误：{e}")
        
        self.status.cache_misses += 1
        return None
    
    async def _cache_set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        if not self.use_cache or not self.redis:
            return
        
        try:
            await self.redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))
            logger.debug(f"缓存已设置：{key}, TTL: {ttl}s")
        except Exception as e:
            logger.warning(f"设置缓存失败：{key}, 错误：{e}")
    
    async def _cache_delete(self, key: str):
        """删除缓存"""
        if not self.use_cache or not self.redis:
            return
        
        try:
            await self.redis.delete(key)
            logger.debug(f"缓存已删除：{key}")
        except Exception as e:
            logger.warning(f"删除缓存失败：{key}, 错误：{e}")
    
    def _compute_hash(self, data: str) -> str:
        """计算数据哈希"""
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_status(self) -> Dict[str, Any]:
        """获取爬虫状态"""
        return self.status.to_dict()
    
    @abstractmethod
    async def crawl(self, **kwargs) -> List[T]:
        """爬虫入口方法（子类必须实现）"""
        pass
    
    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """数据验证方法（子类必须实现）"""
        pass


class DataPipeline(Generic[T]):
    """数据清洗管道"""
    
    def __init__(self):
        self.steps: List[Callable[[Any], Optional[Any]]] = []
    
    def add_step(self, func: Callable[[Any], Optional[Any]]) -> 'DataPipeline':
        """添加清洗步骤"""
        self.steps.append(func)
        return self
    
    def process(self, data: Any) -> Optional[T]:
        """执行清洗流程"""
        result = data
        for step in self.steps:
            try:
                result = step(result)
                if result is None:
                    logger.debug("数据在清洗步骤中被过滤")
                    return None
            except Exception as e:
                logger.warning(f"清洗步骤失败：{step.__name__}, 错误：{e}")
                return None
        return result
    
    def process_batch(self, items: List[Any]) -> List[T]:
        """批量处理数据"""
        results = []
        for item in items:
            result = self.process(item)
            if result is not None:
                results.append(result)
        return results


class DataDeduplicator:
    """数据去重器"""
    
    def __init__(self, capacity: int = 1000000, key_prefix: str = "dedup"):
        self.key_prefix = key_prefix
        self.capacity = capacity
        self.redis: Optional[Redis] = None
    
    async def initialize(self):
        """初始化"""
        try:
            self.redis = get_redis_client()
            await self.redis.ping()
            logger.info("数据去重器初始化成功")
        except Exception as e:
            logger.warning(f"数据去重器初始化失败：{e}")
    
    def _compute_item_hash(self, item: Dict[str, Any], key_fields: List[str]) -> str:
        """计算数据项哈希"""
        key_data = "|".join(str(item.get(field, "")) for field in key_fields)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def is_duplicate(self, item_hash: str) -> bool:
        """检查是否重复"""
        if not self.redis:
            return False
        
        try:
            key = f"{self.key_prefix}:{item_hash}"
            exists = await self.redis.exists(key)
            return bool(exists)
        except Exception as e:
            logger.warning(f"检查重复失败：{e}")
            return False
    
    async def mark_as_seen(self, item_hash: str, ttl: int = 86400):
        """标记为已见"""
        if not self.redis:
            return
        
        try:
            key = f"{self.key_prefix}:{item_hash}"
            await self.redis.setex(key, ttl, "1")
        except Exception as e:
            logger.warning(f"标记重复失败：{e}")
    
    async def filter_duplicates(
        self,
        items: List[Dict[str, Any]],
        key_fields: List[str],
        ttl: int = 86400
    ) -> List[Dict[str, Any]]:
        """过滤重复数据"""
        unique_items = []
        
        for item in items:
            item_hash = self._compute_item_hash(item, key_fields)
            
            if not await self.is_duplicate(item_hash):
                unique_items.append(item)
                await self.mark_as_seen(item_hash, ttl)
        
        logger.info(f"去重：输入{len(items)}条，输出{len(unique_items)}条")
        return unique_items
