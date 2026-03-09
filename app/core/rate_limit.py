"""速率限制中间件

此模块提供API请求速率限制功能。
"""
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict
import logging

from app.core.cache import cache_manager
from app.core.config import config
from app.core.exceptions import RateLimitException

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""
    
    def __init__(
        self,
        app,
        max_requests: int = None,
        window_seconds: int = None
    ):
        super().__init__(app)
        self.max_requests = max_requests or config.RATE_LIMIT_MAX_REQUESTS
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW_SECONDS
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
            return await call_next(request)
        
        if request.url.path.startswith("/static"):
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        rate_limit_key = f"rate_limit:{client_ip}"
        
        current_count = cache_manager.get(rate_limit_key)
        
        if current_count is None:
            cache_manager.set(rate_limit_key, 1, expire=self.window_seconds)
            current_count = 1
        else:
            current_count = int(current_count) + 1
            cache_manager.set(rate_limit_key, current_count, expire=self.window_seconds)
        
        remaining = max(0, self.max_requests - current_count)
        
        if current_count > self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise RateLimitException(f"Rate limit exceeded. Try again in {self.window_seconds} seconds.")
        
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self.window_seconds)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
