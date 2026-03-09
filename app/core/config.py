"""核心配置模块

此模块提供应用程序的配置管理，使用Dynaconf进行配置管理。
"""
import os
from pathlib import Path
from typing import List, Optional
from dynaconf import Dynaconf
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    APP_NAME: str = "PythonDL"
    APP_VERSION: str = "1.0.0"
    APP_DEBUG: bool = False
    
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    SERVER_WORKERS: int = 4
    
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_USER: str = "python"
    DB_PASSWORD: str = "123456"
    DB_NAME: str = "py_demo"
    DB_ECHO: bool = False
    
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "123456"
    REDIS_DB: int = 0
    
    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    LOG_RETENTION_DAYS: int = 30
    
    CACHE_DIR: str = "runtimes/cache"
    CACHE_TTL: int = 3600
    
    TEMP_DIR: str = "temps"
    DATA_DIR: str = "data"
    FILES_DIR: str = "files"
    EXPORT_DIR: str = "files/exports"
    IMAGES_DIR: str = "files/images"
    VIDEOS_DIR: str = "files/videos"
    
    SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    SESSION_EXPIRE_HOURS: int = 24
    PASSWORD_MIN_LENGTH: int = 6
    
    API_PREFIX: str = "/api"
    API_DOCS_URL: str = "/docs"
    API_REDOC_URL: str = "/redoc"
    
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    RATE_LIMIT_MAX_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


BASE_DIR = Path(__file__).resolve().parent.parent.parent

settings = Dynaconf(
    environments=True,
    settings_files=[
        str(BASE_DIR / "settings.toml"),
        str(BASE_DIR / ".secrets.toml"),
    ],
    env_switcher="PYTHONDL_ENV",
    envvar_prefix="PYTHONDL",
    load_dotenv=True,
    dotenv_path=str(BASE_DIR / ".env"),
    root_path=str(BASE_DIR),
)


class Config:
    """配置类，提供便捷的属性访问"""
    
    def get(self, key: str, default=None):
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return settings.get(key, default)
    
    @property
    def APP_NAME(self) -> str:
        return settings.get("app.name", "PythonDL")
    
    @property
    def APP_VERSION(self) -> str:
        return settings.get("app.version", "1.0.0")
    
    @property
    def APP_DEBUG(self) -> bool:
        return settings.get("app.debug", False)
    
    @property
    def SERVER_HOST(self) -> str:
        return settings.get("server.host", "0.0.0.0")
    
    @property
    def SERVER_PORT(self) -> int:
        return settings.get("server.port", 8000)
    
    @property
    def SERVER_WORKERS(self) -> int:
        return settings.get("server.workers", 4)
    
    @property
    def DB_HOST(self) -> str:
        return settings.get("database.host", "127.0.0.1")
    
    @property
    def DB_PORT(self) -> int:
        return settings.get("database.port", 3306)
    
    @property
    def DB_USER(self) -> str:
        return settings.get("database.user", "python")
    
    @property
    def DB_PASSWORD(self) -> str:
        return settings.get("database.password", "123456")
    
    @property
    def DB_NAME(self) -> str:
        return settings.get("database.name", "py_demo")
    
    @property
    def DB_ECHO(self) -> bool:
        return settings.get("database.echo", False)
    
    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"
    
    @property
    def REDIS_HOST(self) -> str:
        return settings.get("redis.host", "127.0.0.1")
    
    @property
    def REDIS_PORT(self) -> int:
        return settings.get("redis.port", 6379)
    
    @property
    def REDIS_PASSWORD(self) -> str:
        return settings.get("redis.password", "123456")
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    @property
    def LOG_DIR(self) -> str:
        return settings.get("log.dir", "logs")
    
    @property
    def LOG_LEVEL(self) -> str:
        return settings.get("log.level", "INFO")
    
    @property
    def LOG_RETENTION_DAYS(self) -> int:
        return settings.get("log.retention_days", 30)
    
    @property
    def CACHE_DIR(self) -> str:
        return settings.get("cache.dir", "runtimes/cache")
    
    @property
    def CACHE_TTL(self) -> int:
        return settings.get("cache.ttl", 3600)
    
    @property
    def TEMP_DIR(self) -> str:
        return settings.get("temp.dir", "temps")
    
    @property
    def DATA_DIR(self) -> str:
        return settings.get("data.dir", "data")
    
    @property
    def FILES_DIR(self) -> str:
        return settings.get("files.dir", "files")
    
    @property
    def EXPORT_DIR(self) -> str:
        return settings.get("files.export_dir", "files/exports")
    
    @property
    def IMAGES_DIR(self) -> str:
        return settings.get("files.images_dir", "files/images")
    
    @property
    def VIDEOS_DIR(self) -> str:
        return settings.get("files.videos_dir", "files/videos")
    
    @property
    def SECRET_KEY(self) -> str:
        return settings.get("security.secret_key", "your-secret-key-change-in-production")
    
    @property
    def JWT_SECRET_KEY(self) -> str:
        return settings.get("security.jwt_secret_key", "jwt-secret-key-change-in-production")
    
    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return settings.get("security.access_token_expire_minutes", 30)
    
    @property
    def REFRESH_TOKEN_EXPIRE_DAYS(self) -> int:
        return settings.get("security.refresh_token_expire_days", 7)
    
    @property
    def SESSION_EXPIRE_HOURS(self) -> int:
        return settings.get("security.session_expire_hours", 24)
    
    @property
    def PASSWORD_MIN_LENGTH(self) -> int:
        return settings.get("security.password_min_length", 6)
    
    @property
    def API_PREFIX(self) -> str:
        return settings.get("api.prefix", "/api")
    
    @property
    def API_DOCS_URL(self) -> str:
        return settings.get("api.docs_url", "/docs")
    
    @property
    def API_REDOC_URL(self) -> str:
        return settings.get("api.redoc_url", "/redoc")
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        return settings.get("cors.origins", ["*"])
    
    @property
    def CORS_ALLOW_CREDENTIALS(self) -> bool:
        return settings.get("cors.allow_credentials", True)
    
    @property
    def CORS_ALLOW_METHODS(self) -> List[str]:
        return settings.get("cors.allow_methods", ["*"])
    
    @property
    def CORS_ALLOW_HEADERS(self) -> List[str]:
        return settings.get("cors.allow_headers", ["*"])
    
    @property
    def RATE_LIMIT_MAX_REQUESTS(self) -> int:
        return settings.get("rate_limit.max_requests", 1000)
    
    @property
    def RATE_LIMIT_WINDOW_SECONDS(self) -> int:
        return settings.get("rate_limit.window_seconds", 60)


config = Config()
