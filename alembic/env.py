"""Alembic配置文件

此模块用于数据库迁移管理。
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import Base
from app.models import User, Role, Permission, RolePermission, Stock, StockBasic, Weather, FaceReading, FortuneTelling, ConsumptionData, EconomicIndicator, DataSource, Analysis
from app.core.config import config

# Alembic配置对象
config = context.config

# 解释配置文件的Python日志记录
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置meta数据标识符
target_metadata = Base.metadata

def run_migrations_offline():
    """在'离线'模式下运行迁移
    
    这将配置上下文，只需URL，而不是引擎。
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"}
    )
    
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """在'在线'模式下运行迁移
    
    在这种情况下，我们需要创建引擎并将其与上下文关联。
    """
    # 从配置创建引擎
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        
        with context.begin_transaction():
            context.run_migrations()