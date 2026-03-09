"""数据库连接模块

此模块提供数据库连接和会话管理。
"""
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging

from app.core.config import config

logger = logging.getLogger(__name__)

engine = create_engine(
    config.DATABASE_URL,
    echo=config.DB_ECHO,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
    poolclass=QueuePool,
    connect_args={
        "charset": "utf8mb4",
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话
    
    Yields:
        Session: 数据库会话
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """获取数据库会话上下文管理器
    
    Yields:
        Session: 数据库会话
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@event.listens_for(engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    """数据库连接事件处理"""
    logger.debug("Database connection established")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """数据库连接检出事件处理"""
    logger.debug("Database connection checked out")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """数据库连接归还事件处理"""
    logger.debug("Database connection checked in")
