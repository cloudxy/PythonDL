"""API响应模块

此模块提供统一的API响应格式。
"""
from typing import Any, Optional, Dict
from pydantic import BaseModel


class ApiResponse:
    """统一API响应类"""
    
    @staticmethod
    def success(data: Any = None, message: str = "操作成功") -> Dict[str, Any]:
        """成功响应
        
        Args:
            data: 响应数据
            message: 响应消息
            
        Returns:
            Dict[str, Any]: 响应字典
        """
        return {
            "success": True,
            "data": data,
            "message": message
        }
    
    @staticmethod
    def error(message: str = "操作失败", code: int = 400, data: Any = None) -> Dict[str, Any]:
        """错误响应
        
        Args:
            message: 错误消息
            code: 错误码
            data: 附加数据
            
        Returns:
            Dict[str, Any]: 响应字典
        """
        return {
            "success": False,
            "data": data,
            "message": message,
            "code": code
        }
    
    @staticmethod
    def paginated(
        items: list,
        total: int,
        page: int = 1,
        page_size: int = 20,
        message: str = "获取成功"
    ) -> Dict[str, Any]:
        """分页响应
        
        Args:
            items: 数据列表
            total: 总数
            page: 当前页
            page_size: 每页大小
            message: 响应消息
            
        Returns:
            Dict[str, Any]: 响应字典
        """
        return {
            "success": True,
            "data": {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0
            },
            "message": message
        }


class ResponseModel(BaseModel):
    """响应模型"""
    success: bool = True
    data: Optional[Any] = None
    message: str = "操作成功"
    code: Optional[int] = None
