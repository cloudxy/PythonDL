"""监控模块

此模块提供性能监控和请求上下文管理功能。
"""
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

from app.core.logger import get_logger

logger = get_logger("monitoring")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """请求上下文中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        process_time_ms = round(process_time * 1000, 2)
        
        response.headers["X-Process-Time"] = f"{process_time_ms}ms"
        
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time_ms}ms"
            )
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time_ms}ms"
        )
        
        return response
