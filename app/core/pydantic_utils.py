"""Pydantic 工具模块

此模块提供 Pydantic v2 兼容工具函数。
"""
from typing import TypeVar, Type, Any, List

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


def from_orm(model: Type[T], obj: Any) -> T:
    """Pydantic v2 兼容的 from_orm 方法
    
    Args:
        model: Pydantic 模型类
        obj: ORM 对象或其他数据源
        
    Returns:
        T: Pydantic 模型实例
    """
    if hasattr(model, 'model_validate'):
        return model.model_validate(obj)  # type: ignore
    else:
        return model.from_orm(obj)  # type: ignore


def from_orm_list(model: Type[T], items: List[Any]) -> List[T]:
    """Pydantic v2 兼容的 from_orm 列表方法
    
    Args:
        model: Pydantic 模型类
        items: ORM 对象列表
        
    Returns:
        List[T]: Pydantic 模型实例列表
    """
    return [from_orm(model, item) for item in items]
