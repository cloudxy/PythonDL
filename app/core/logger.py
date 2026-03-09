"""日志系统模块

此模块提供统一的日志记录功能，支持按天流转，区分普通日志和错误日志。
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from typing import Optional
import os

from app.core.config import config


class CustomFormatter(logging.Formatter):
    """自定义日志格式化器"""
    
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_str)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    """按天流转的日志处理器"""
    
    def __init__(self, filename, backupCount=30, encoding='utf-8'):
        super().__init__(
            filename=filename,
            when='midnight',
            interval=1,
            backupCount=backupCount,
            encoding=encoding
        )
        self.suffix = "%Y-%m-%d"
    
    def getFilesToDelete(self):
        """获取需要删除的日志文件"""
        dir_name, base_name = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name)
        result = []
        for file_name in file_names:
            if file_name.startswith(base_name):
                result.append(os.path.join(dir_name, file_name))
        
        if len(result) < self.backupCount:
            return []
        
        result.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return result[self.backupCount:]


def setup_logging(log_level: Optional[str] = None):
    """设置日志系统
    
    Args:
        log_level: 日志级别
    """
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    app_log_dir = log_dir / "app"
    error_log_dir = log_dir / "error"
    gunicorn_log_dir = log_dir / "gunicorn"
    
    app_log_dir.mkdir(parents=True, exist_ok=True)
    error_log_dir.mkdir(parents=True, exist_ok=True)
    gunicorn_log_dir.mkdir(parents=True, exist_ok=True)
    
    level = getattr(logging, log_level or config.LOG_LEVEL)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)
    
    app_handler = DailyRotatingFileHandler(
        filename=str(app_log_dir / "app.log"),
        backupCount=config.LOG_RETENTION_DAYS
    )
    app_handler.setLevel(level)
    app_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    ))
    root_logger.addHandler(app_handler)
    
    error_handler = DailyRotatingFileHandler(
        filename=str(error_log_dir / "error.log"),
        backupCount=config.LOG_RETENTION_DAYS
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n"
        "Exception: %(exc_info)s\n"
    ))
    root_logger.addHandler(error_handler)
    
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)


logger = setup_logging()
