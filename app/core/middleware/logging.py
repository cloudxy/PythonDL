"""API日志记录中间件

此模块提供用于记录API请求和响应的装饰器。
"""
import time
import functools
from typing import Optional, Callable, Any
from fastapi import Request, Response
from app.core.logger import get_logger
from app.core.auth import verify_token

logger = get_logger("api_logging")


def api_logger(level: str = "info", module: Optional[str] = None, action: Optional[str] = None):
    """
    API日志记录装饰器
    
    Args:
        level: 日志级别，默认为"info"
        module: 模块名，默认为None
        action: 操作名，默认为None
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request: Optional[Request] = None
            db = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if "db" in kwargs:
                db = kwargs["db"]
            elif "session" in kwargs:
                db = kwargs["session"]
            
            user_id = None
            username = "未知用户"
            
            if request:
                authorization = request.headers.get("Authorization")
                if authorization and authorization.startswith("Bearer "):
                    token = authorization.split(" ")[1]
                    payload = verify_token(token)
                    if payload:
                        user_id = payload.get("sub")
                        if user_id and db:
                            from app.models.admin.user import User
                            user = db.query(User).filter(User.id == user_id).first()
                            if user:
                                username = user.username
            
            if not module:
                module_name = func.__module__.split(".")[-1]
            else:
                module_name = module
            
            if not action:
                action_name = func.__name__
            else:
                action_name = action
            
            start_time = time.time()
            request_info = {}
            
            if request:
                request_info = {
                    "method": request.method,
                    "url": str(request.url),
                }
                try:
                    body = await request.body()
                    if body:
                        request_info["body"] = body.decode("utf-8")
                except Exception as e:
                    logger.debug(f"Failed to get request body: {str(e)}")
            
            logger.debug(f"API request started: {action_name} for user {username}")
            
            try:
                result = await func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                response_info = {}
                if isinstance(result, Response):
                    response_info = {
                        "status_code": result.status_code,
                    }
                else:
                    response_info = {
                        "status": "success",
                    }
                
                logger_message = f"API request successful: {action_name} for user {username} in {execution_time:.4f}s"
                if level == "info":
                    logger.info(logger_message)
                elif level == "debug":
                    logger.debug(logger_message)
                elif level == "warning":
                    logger.warning(logger_message)
                elif level == "error":
                    logger.error(logger_message)
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                error_message = f"API request failed: {action_name} for user {username} in {execution_time:.4f}s - {str(e)}"
                logger.error(error_message, exc_info=True)
                
                raise
        
        return wrapper
    
    return decorator


class LoggingMiddleware:
    """
    FastAPI日志记录中间件
    
    用于记录所有API请求和响应的详细信息。
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        request = Request(scope, receive)
        
        logger.info(f"Request started: {request.method} {request.url}")
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                execution_time = time.time() - start_time
                logger.info(f"Request completed: {request.method} {request.url} - {status_code} in {execution_time:.4f}s")
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Request failed: {request.method} {request.url} - {str(e)} in {execution_time:.4f}s", exc_info=True)
            raise
