"""异常处理模块

此模块提供统一的异常处理功能。
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from typing import Union
import logging

logger = logging.getLogger(__name__)


class AppException(Exception):
    """应用异常基类"""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR"
    ):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.detail)


class NotFoundException(AppException):
    """资源未找到异常"""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="NOT_FOUND"
        )


class BadRequestException(AppException):
    """错误请求异常"""
    
    def __init__(self, detail: str = "Bad request"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="BAD_REQUEST"
        )


class UnauthorizedException(AppException):
    """未授权异常"""
    
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED"
        )


class ForbiddenException(AppException):
    """禁止访问异常"""
    
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN"
        )


class ConflictException(AppException):
    """冲突异常"""
    
    def __init__(self, detail: str = "Conflict"):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT"
        )


class ValidationException(AppException):
    """验证异常"""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )


class RateLimitException(AppException):
    """速率限制异常"""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED"
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """应用异常处理器"""
    logger.error(f"App exception: {exc.error_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.detail
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """验证异常处理器"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"]
        })
    
    logger.error(f"Validation error: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Validation error",
            "errors": errors
        }
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """数据库异常处理器"""
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "DATABASE_ERROR",
            "message": "Database error occurred"
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred"
        }
    )


def setup_exception_handlers(app):
    """设置异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
