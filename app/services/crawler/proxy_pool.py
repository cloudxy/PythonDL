"""智能代理池管理

此模块提供代理 IP 的获取、验证、轮询等功能，防止爬虫被封禁。
"""
import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

import aiohttp

from app.core.logger import get_logger

logger = get_logger("proxy_pool")


class ProxyStatus(Enum):
    """代理状态枚举"""
    AVAILABLE = "available"
    TESTING = "testing"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Proxy:
    """代理 IP 数据类"""
    host: str
    port: int
    protocol: str = "http"
    status: ProxyStatus = ProxyStatus.AVAILABLE
    country: str = ""
    anonymity: str = "high"  # high, anonymous, transparent
    speed: float = 0.0  # 响应时间（秒）
    success_rate: float = 1.0  # 成功率
    last_checked: Optional[datetime] = None
    last_used: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    
    @property
    def url(self) -> str:
        """获取代理 URL"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.status == ProxyStatus.AVAILABLE
    
    def mark_success(self):
        """标记成功"""
        self.success_count += 1
        self.failure_count = 0
        self.last_checked = datetime.now()
        self.last_used = datetime.now()
        self.status = ProxyStatus.AVAILABLE
        self._update_success_rate()
    
    def mark_failure(self):
        """标记失败"""
        self.failure_count += 1
        self.last_checked = datetime.now()
        self._update_success_rate()
        
        if self.failure_count >= 3:
            self.status = ProxyStatus.FAILED
            logger.warning(f"代理已标记为失败：{self.url}")
    
    def mark_blocked(self):
        """标记被封禁"""
        self.status = ProxyStatus.BLOCKED
        logger.warning(f"代理已标记为被封禁：{self.url}")
    
    def _update_success_rate(self):
        """更新成功率"""
        total = self.success_count + self.failure_count
        if total > 0:
            self.success_rate = self.success_count / total


class ProxyPool:
    """代理池管理器"""
    
    def __init__(
        self,
        max_size: int = 100,
        test_interval: int = 300,  # 5 分钟
        test_timeout: int = 10,
        min_success_rate: float = 0.6
    ):
        self.max_size = max_size
        self.test_interval = test_interval
        self.test_timeout = test_timeout
        self.min_success_rate = min_success_rate
        
        # 代理存储
        self.proxies: Dict[str, Proxy] = {}
        self.available_proxies: List[str] = []
        self.current_index: int = 0
        
        # 锁定
        self.lock = asyncio.Lock()
        
        # 测试会话
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 测试 URL（用于验证代理是否可用）
        self.test_urls = [
            "https://httpbin.org/ip",
            "https://api.ip.sb/ip",
            "https://ifconfig.me/ip"
        ]
        
        # 代理源（可以扩展更多）
        self.proxy_sources = [
            "https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all",
            "https://www.proxy-list.download/api/v1/get?type=http",
        ]
    
    async def initialize(self):
        """初始化代理池"""
        timeout = aiohttp.ClientTimeout(total=self.test_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("代理池初始化完成")
    
    async def close(self):
        """关闭代理池"""
        if self.session:
            await self.session.close()
            logger.info("代理池已关闭")
    
    async def add_proxy(self, proxy: Proxy) -> bool:
        """添加代理到池中"""
        async with self.lock:
            key = proxy.url
            
            if key in self.proxies:
                logger.debug(f"代理已存在：{key}")
                return False
            
            if len(self.proxies) >= self.max_size:
                # 池满，移除最旧的可用代理
                if self.available_proxies:
                    oldest_key = self.available_proxies.pop(0)
                    del self.proxies[oldest_key]
                    logger.debug(f"移除最旧代理：{oldest_key}")
            
            self.proxies[key] = proxy
            if proxy.is_available:
                self.available_proxies.append(key)
            
            logger.info(f"添加代理：{key}, 当前池大小：{len(self.proxies)}")
            return True
    
    async def add_proxies_from_list(self, proxy_list: List[str], protocol: str = "http"):
        """批量添加代理"""
        for proxy_str in proxy_list:
            try:
                parts = proxy_str.split(':')
                if len(parts) >= 2:
                    host = parts[0]
                    port = int(parts[1])
                    proxy = Proxy(host=host, port=port, protocol=protocol)
                    await self.add_proxy(proxy)
            except Exception as e:
                logger.warning(f"解析代理失败：{proxy_str}, 错误：{e}")
    
    async def get_proxy(self) -> Optional[str]:
        """获取一个可用代理"""
        async with self.lock:
            if not self.available_proxies:
                logger.warning("代理池为空")
                return None
            
            # 轮询选择代理
            for _ in range(len(self.available_proxies)):
                key = self.available_proxies[self.current_index % len(self.available_proxies)]
                self.current_index += 1
                
                proxy = self.proxies.get(key)
                if proxy and proxy.is_available:
                    proxy.last_used = datetime.now()
                    logger.debug(f"使用代理：{proxy.url}")
                    return proxy.url
            
            logger.warning("没有可用的代理")
            return None
    
    async def remove_proxy(self, proxy_url: str):
        """移除代理"""
        async with self.lock:
            if proxy_url in self.proxies:
                proxy = self.proxies[proxy_url]
                proxy.status = ProxyStatus.FAILED
                
                if proxy_url in self.available_proxies:
                    self.available_proxies.remove(proxy_url)
                
                logger.info(f"移除代理：{proxy_url}")
    
    async def mark_proxy_failed(self, proxy_url: str):
        """标记代理失败"""
        async with self.lock:
            if proxy_url in self.proxies:
                proxy = self.proxies[proxy_url]
                proxy.mark_failure()
                
                if proxy.status == ProxyStatus.FAILED and proxy_url in self.available_proxies:
                    self.available_proxies.remove(proxy_url)
    
    async def mark_proxy_success(self, proxy_url: str):
        """标记代理成功"""
        async with self.lock:
            if proxy_url in self.proxies:
                proxy = self.proxies[proxy_url]
                proxy.mark_success()
                
                if proxy_url not in self.available_proxies and proxy.is_available:
                    self.available_proxies.append(proxy_url)
    
    async def test_proxy(self, proxy: Proxy) -> bool:
        """测试代理是否可用"""
        try:
            proxy.status = ProxyStatus.TESTING
            
            connector = aiohttp.TCPConnector()
            async with aiohttp.ClientSession(connector=connector) as session:
                start_time = time.time()
                
                async with session.get(
                    random.choice(self.test_urls),
                    proxy=proxy.url,
                    timeout=aiohttp.ClientTimeout(total=self.test_timeout)
                ) as response:
                    if response.status == 200:
                        elapsed = time.time() - start_time
                        proxy.speed = elapsed
                        proxy.mark_success()
                        logger.info(f"代理测试成功：{proxy.url}, 速度：{elapsed:.2f}s")
                        return True
                    else:
                        proxy.mark_failure()
                        logger.warning(f"代理测试失败：{proxy.url}, 状态码：{response.status}")
                        return False
                        
        except Exception as e:
            proxy.mark_failure()
            logger.warning(f"代理测试异常：{proxy.url}, 错误：{e}")
            return False
    
    async def test_all_proxies(self):
        """测试所有代理"""
        logger.info(f"开始测试所有代理，总数：{len(self.proxies)}")
        
        tasks = []
        for proxy in self.proxies.values():
            if proxy.status != ProxyStatus.BLOCKED:
                tasks.append(self.test_proxy(proxy))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.info(f"代理测试完成，成功：{success_count}/{len(self.proxies)}")
    
    async def fetch_from_sources(self):
        """从代理源获取代理"""
        logger.info("从代理源获取代理")
        
        for source_url in self.proxy_sources:
            try:
                async with self.session.get(source_url) as response:
                    if response.status == 200:
                        text = await response.text()
                        proxy_list = text.strip().split('\n')
                        await self.add_proxies_from_list(proxy_list)
                        logger.info(f"从源获取代理成功：{source_url}, 数量：{len(proxy_list)}")
            except Exception as e:
                logger.error(f"从源获取代理失败：{source_url}, 错误：{e}")
    
    def get_stats(self) -> Dict[str, any]:
        """获取代理池统计"""
        total = len(self.proxies)
        available = len(self.available_proxies)
        failed = sum(1 for p in self.proxies.values() if p.status == ProxyStatus.FAILED)
        blocked = sum(1 for p in self.proxies.values() if p.status == ProxyStatus.BLOCKED)
        
        avg_speed = sum(p.speed for p in self.proxies.values() if p.speed > 0) / max(1, total)
        avg_success_rate = sum(p.success_rate for p in self.proxies.values()) / max(1, total)
        
        return {
            "total": total,
            "available": available,
            "failed": failed,
            "blocked": blocked,
            "avg_speed": round(avg_speed, 2),
            "avg_success_rate": round(avg_success_rate, 2),
        }


class ProxyMiddleware:
    """代理中间件（用于自动切换代理）"""
    
    def __init__(self, proxy_pool: ProxyPool):
        self.proxy_pool = proxy_pool
        self.max_retries = 3
    
    async def request_with_proxy(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        **kwargs
    ) -> Optional[aiohttp.ClientResponse]:
        """带代理的请求（自动切换）"""
        for attempt in range(self.max_retries):
            proxy_url = await self.proxy_pool.get_proxy()
            
            if not proxy_url:
                # 没有代理，直接请求
                logger.warning("没有代理，使用直接连接")
                async with session.request(method, url, **kwargs) as response:
                    return response
            
            try:
                logger.debug(f"使用代理请求：{url}, 代理：{proxy_url}")
                async with session.request(method, url, proxy=proxy_url, **kwargs) as response:
                    if response.status == 403:
                        # 被封禁，标记代理并切换
                        logger.warning(f"请求被拒绝 (403)，切换代理：{url}")
                        await self.proxy_pool.mark_proxy_failed(proxy_url)
                        continue
                    
                    await self.proxy_pool.mark_proxy_success(proxy_url)
                    return response
                    
            except Exception as e:
                logger.warning(f"代理请求失败：{url}, 代理：{proxy_url}, 错误：{e}")
                await self.proxy_pool.mark_proxy_failed(proxy_url)
                
                if attempt == self.max_retries - 1:
                    logger.error(f"代理请求最终失败：{url}")
                    raise
        
        return None


# 全局代理池实例
_proxy_pool: Optional[ProxyPool] = None


async def get_proxy_pool() -> ProxyPool:
    """获取代理池实例"""
    global _proxy_pool
    
    if _proxy_pool is None:
        _proxy_pool = ProxyPool()
        await _proxy_pool.initialize()
    
    return _proxy_pool


async def close_proxy_pool():
    """关闭代理池"""
    global _proxy_pool
    
    if _proxy_pool:
        await _proxy_pool.close()
        _proxy_pool = None
