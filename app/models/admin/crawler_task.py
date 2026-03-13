"""爬虫任务管理数据模型

此模块定义爬虫任务和日志数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index, JSON
from sqlalchemy import Enum as SQLEnum
import enum

from app.core.database import Base


class TaskStatus(enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"  # 待执行
    RUNNING = "running"  # 执行中
    SUCCESS = "success"  # 成功
    FAILED = "failed"  # 失败
    STOPPED = "stopped"  # 已停止


class CrawlerTask(Base):
    """爬虫任务模型"""
    
    __tablename__ = "crawler_tasks"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True, comment="任务名称")
    task_type = Column(String(50), nullable=False, index=True, comment="任务类型")
    description = Column(Text, nullable=True, comment="任务描述")
    
    # 爬虫配置
    crawler_config = Column(JSON, nullable=True, comment="爬虫配置")
    target_urls = Column(Text, nullable=True, comment="目标 URLs（多行）")
    parse_rules = Column(JSON, nullable=True, comment="解析规则")
    
    # 调度配置
    schedule_type = Column(String(20), default="manual", comment="调度类型：manual, interval, cron")
    cron_expression = Column(String(100), nullable=True, comment="Cron 表达式")
    interval_seconds = Column(Integer, default=3600, comment="间隔秒数")
    
    # 任务状态
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True, comment="任务状态")
    last_run_at = Column(DateTime, nullable=True, comment="最后执行时间")
    next_run_at = Column(DateTime, nullable=True, comment="下次执行时间")
    
    # 执行统计
    total_runs = Column(Integer, default=0, comment="总执行次数")
    success_runs = Column(Integer, default=0, comment="成功执行次数")
    failed_runs = Column(Integer, default=0, comment="失败执行次数")
    total_records = Column(Integer, default=0, comment="总采集记录数")
    
    # 控制字段
    is_active = Column(Boolean, default=True, index=True, comment="是否启用")
    is_running = Column(Boolean, default=False, comment="是否正在运行")
    
    # 元数据
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True, comment="创建人 ID")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    __table_args__ = (
        Index("idx_task_type_status", "task_type", "status"),
    )


class CrawlerLog(Base):
    """爬虫日志模型"""
    
    __tablename__ = "crawler_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("crawler_tasks.id"), nullable=False, index=True, comment="任务 ID")
    task_name = Column(String(200), index=True, comment="任务名称")
    
    # 日志级别
    log_level = Column(String(20), default="INFO", index=True, comment="日志级别")
    
    # 日志内容
    message = Column(Text, nullable=False, comment="日志消息")
    details = Column(JSON, nullable=True, comment="详细信息")
    
    # 执行信息
    execution_id = Column(String(100), index=True, comment="执行 ID")
    start_time = Column(DateTime, nullable=True, comment="开始时间")
    end_time = Column(DateTime, nullable=True, comment="结束时间")
    duration_seconds = Column(Float, nullable=True, comment="耗时（秒）")
    
    # 采集统计
    records_fetched = Column(Integer, default=0, comment="采集记录数")
    records_saved = Column(Integer, default=0, comment="保存记录数")
    error_count = Column(Integer, default=0, comment="错误数")
    
    # 错误信息
    error_message = Column(Text, nullable=True, comment="错误信息")
    stack_trace = Column(Text, nullable=True, comment="堆栈跟踪")
    
    # 元数据
    created_at = Column(DateTime, default=datetime.now, index=True, comment="创建时间")
    
    __table_args__ = (
        Index("idx_task_level_time", "task_id", "log_level", "created_at"),
    )


class CrawlerData(Base):
    """爬虫采集数据模型（通用）"""
    
    __tablename__ = "crawler_data"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("crawler_tasks.id"), nullable=False, index=True, comment="任务 ID")
    source_type = Column(String(50), nullable=False, index=True, comment="数据源类型")
    source_url = Column(String(500), nullable=True, comment="来源 URL")
    
    # 数据内容（JSON 格式存储）
    data_content = Column(JSON, nullable=False, comment="数据内容")
    data_hash = Column(String(64), index=True, comment="数据哈希（去重）")
    
    # 处理状态
    is_processed = Column(Boolean, default=False, index=True, comment="是否已处理")
    processed_at = Column(DateTime, nullable=True, comment="处理时间")
    
    # 元数据
    created_at = Column(DateTime, default=datetime.now, index=True, comment="采集时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    __table_args__ = (
        Index("idx_source_type_time", "source_type", "created_at"),
    )
