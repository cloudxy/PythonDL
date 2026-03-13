"""安全模块

此模块提供密码加密、验证等安全相关功能。
"""
import secrets
import hashlib
from typing import Optional
import bcrypt
import logging

from app.core.config import config

logger = logging.getLogger(__name__)


def get_password_hash(password: str) -> str:
    """生成密码哈希
    
    Args:
        password: 明文密码
        
    Returns:
        str: 密码哈希
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码
    
    Args:
        plain_password: 明文密码
        hashed_password: 密码哈希
        
    Returns:
        bool: 是否验证通过
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """验证密码强度
    
    Args:
        password: 密码
        
    Returns:
        tuple[bool, str]: (是否通过, 错误信息)
    """
    min_length = config.PASSWORD_MIN_LENGTH
    
    if len(password) < min_length:
        return False, f"密码长度不能少于{min_length}个字符"
    
    if not any(c.isupper() for c in password):
        return False, "密码必须包含至少一个大写字母"
    
    if not any(c.islower() for c in password):
        return False, "密码必须包含至少一个小写字母"
    
    if not any(c.isdigit() for c in password):
        return False, "密码必须包含至少一个数字"
    
    return True, "密码强度验证通过"


def generate_random_token(length: int = 32) -> str:
    """生成随机令牌
    
    Args:
        length: 令牌长度
        
    Returns:
        str: 随机令牌
    """
    return secrets.token_hex(length)


def generate_api_key() -> str:
    """生成API密钥
    
    Returns:
        str: API密钥
    """
    return f"pk_{secrets.token_urlsafe(32)}"


def hash_string(text: str) -> str:
    """生成字符串哈希
    
    Args:
        text: 原始字符串
        
    Returns:
        str: 哈希值
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def generate_password_reset_token(email: str) -> str:
    """生成密码重置令牌
    
    Args:
        email: 用户邮箱
        
    Returns:
        str: 重置令牌
    """
    return secrets.token_urlsafe(32)


def verify_password_reset_token(token: str) -> Optional[str]:
    """验证密码重置令牌
    
    Args:
        token: 重置令牌
        
    Returns:
        Optional[str]: 验证成功返回邮箱，失败返回 None
    """
    from app.core.cache import cache_manager
    
    cache_key = f"password_reset:{token}"
    email = cache_manager.get(cache_key)
    
    if email:
        cache_manager.delete(cache_key)
    
    return email


# 从 auth 模块导入认证相关函数（兼容旧代码）
try:
    from app.core.auth import get_current_user, create_access_token, verify_token  # noqa
except ImportError:
    # 如果 auth 模块不可用，使用占位函数
    async def get_current_user(token: str = ""):
        """占位函数"""
        pass
    
    def create_access_token(data: dict):
        """占位函数"""
        return "token"
    
    def verify_token(token: str):
        """占位函数"""
        return None
