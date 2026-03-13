"""爬虫任务管理 Schema

此模块定义爬虫任务管理相关的数据验证 Schema。
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class TaskStatusEnum(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class CrawlerTaskBase(BaseModel):
    """爬虫任务基础 Schema"""
    name: str = Field(..., description="任务名称", min_length=1, max_length=200)
    task_type: str = Field(..., description="任务类型", max_length=50)
    description: Optional[str] = Field(None, description="任务描述")
    crawler_config: Optional[Dict[str, Any]] = Field(None, description="爬虫配置")
    target_urls: Optional[str] = Field(None, description="目标 URLs")
    parse_rules: Optional[Dict[str, Any]] = Field(None, description="解析规则")
    schedule_type: str = Field("manual", description="调度类型")
    cron_expression: Optional[str] = Field(None, description="Cron 表达式")
    interval_seconds: int = Field(3600, description="间隔秒数", ge=60)


class CrawlerTaskCreate(CrawlerTaskBase):
    """创建爬虫任务 Schema"""
    pass


class CrawlerTaskUpdate(BaseModel):
    """更新爬虫任务 Schema"""
    name: Optional[str] = Field(None, description="任务名称", min_length=1, max_length=200)
    task_type: Optional[str] = Field(None, description="任务类型", max_length=50)
    description: Optional[str] = Field(None, description="任务描述")
    crawler_config: Optional[Dict[str, Any]] = Field(None, description="爬虫配置")
    target_urls: Optional[str] = Field(None, description="目标 URLs")
    parse_rules: Optional[Dict[str, Any]] = Field(None, description="解析规则")
    schedule_type: Optional[str] = Field(None, description="调度类型")
    cron_expression: Optional[str] = Field(None, description="Cron 表达式")
    interval_seconds: Optional[int] = Field(None, description="间隔秒数", ge=60)
    is_active: Optional[bool] = Field(None, description="是否启用")


class CrawlerTaskResponse(CrawlerTaskBase):
    """爬虫任务响应 Schema"""
    id: int
    status: Optional[TaskStatusEnum] = None
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    total_runs: int = 0
    success_runs: int = 0
    failed_runs: int = 0
    total_records: int = 0
    is_active: bool = True
    is_running: bool = False
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CrawlerTaskList(BaseModel):
    """爬虫任务列表响应 Schema"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


class CrawlerLogBase(BaseModel):
    """日志基础 Schema"""
    task_id: int = Field(..., description="任务 ID")
    log_level: str = Field("INFO", description="日志级别", max_length=20)
    message: str = Field(..., description="日志消息")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")


class CrawlerLogResponse(CrawlerLogBase):
    """日志响应 Schema"""
    id: int
    task_name: str
    execution_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    records_fetched: int = 0
    records_saved: int = 0
    error_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class CrawlerLogList(BaseModel):
    """日志列表响应 Schema"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


class CrawlerDataResponse(BaseModel):
    """采集数据响应 Schema"""
    id: int
    task_id: int
    source_type: str
    source_url: Optional[str] = None
    data_content: Dict[str, Any]
    is_processed: bool = False
    processed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
