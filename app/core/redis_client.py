"""Redis 缓存客户端

此模块提供 Redis 缓存连接和操作功能。
"""
import json
import logging
from typing import Any, Optional, List, Union
from datetime import timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from app.core.config import config

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 客户端类"""
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.is_initialized = False
    
    async def initialize(self):
        """初始化 Redis 连接池"""
        if not self.is_initialized:
            try:
                self.pool = ConnectionPool(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    password=config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
                    db=0,
                    decode_responses=True,
                    max_connections=50,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True
                )
                
                self.client = redis.Redis(connection_pool=self.pool)
                self.is_initialized = True
                
                # 测试连接
                await self.client.ping()
                logger.info("Redis 连接成功")
                
            except Exception as e:
                logger.error(f"Redis 连接失败：{e}")
                self.is_initialized = False
    
    async def close(self):
        """关闭 Redis 连接"""
        if self.pool:
            await self.pool.disconnect()
            self.is_initialized = False
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            value = await self.client.get(key)
            if value:
                # 尝试 JSON 解析
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except Exception as e:
            logger.error(f"Redis GET 失败 {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 序列化值
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if expire:
                await self.client.setex(key, expire, value)
            else:
                await self.client.set(key, value)
                
        except Exception as e:
            logger.error(f"Redis SET 失败 {key}: {e}")
    
    async def delete(self, key: Union[str, List[str]]) -> int:
        """删除缓存
        
        Args:
            key: 缓存键或键列表
            
        Returns:
            删除的键数量
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if isinstance(key, list):
                return await self.client.delete(*key)
            else:
                return await self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE 失败 {key}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS 失败 {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置键的过期时间"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Redis EXPIRE 失败 {key}: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """自增操作"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.incr(key, amount)
        except Exception as e:
            logger.error(f"Redis INCR 失败 {key}: {e}")
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """自减操作"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.decr(key, amount)
        except Exception as e:
            logger.error(f"Redis DECR 失败 {key}: {e}")
            return 0
    
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """获取 Hash 字段值"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            value = await self.client.hget(name, key)
            if value:
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except Exception as e:
            logger.error(f"Redis HGET 失败 {name}.{key}: {e}")
            return None
    
    async def hset(self, name: str, key: str, value: Any):
        """设置 Hash 字段值"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await self.client.hset(name, key, value)
        except Exception as e:
            logger.error(f"Redis HSET 失败 {name}.{key}: {e}")
    
    async def hgetall(self, name: str) -> dict:
        """获取整个 Hash"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.hgetall(name)
        except Exception as e:
            logger.error(f"Redis HGETALL 失败 {name}: {e}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """删除 Hash 字段"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Redis HDEL 失败 {name}: {e}")
            return 0
    
    async def lpush(self, name: str, *values: Any):
        """列表左侧推入"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            for value in values:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                await self.client.lpush(name, value)
        except Exception as e:
            logger.error(f"Redis LPUSH 失败 {name}: {e}")
    
    async def rpush(self, name: str, *values: Any):
        """列表右侧推入"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            for value in values:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                await self.client.rpush(name, value)
        except Exception as e:
            logger.error(f"Redis RPUSH 失败 {name}: {e}")
    
    async def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """获取列表范围"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            values = await self.client.lrange(name, start, end)
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except:
                    result.append(v)
            return result
        except Exception as e:
            logger.error(f"Redis LRANGE 失败 {name}: {e}")
            return []
    
    async def keys(self, pattern: str) -> List[str]:
        """获取匹配模式的键"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS 失败 {pattern}: {e}")
            return []
    
    async def flushdb(self):
        """清空当前数据库"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            await self.client.flushdb()
            logger.warning("Redis 数据库已清空")
        except Exception as e:
            logger.error(f"Redis FLUSHDB 失败：{e}")
    
    async def ping(self) -> bool:
        """测试连接"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error(f"Redis PING 失败：{e}")
            return False


# 全局 Redis 客户端实例
redis_client = RedisClient()


async def get_redis_client() -> RedisClient:
    """获取 Redis 客户端实例"""
    if not redis_client.is_initialized:
        await redis_client.initialize()
    return redis_client
