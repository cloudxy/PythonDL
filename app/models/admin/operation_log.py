"""操作日志模型

此模块定义操作日志数据模型。
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from app.core.database import Base


class OperationLog(Base):
    """操作日志模型"""
    
    __tablename__ = "operation_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True, comment="用户ID")
    username = Column(String(50), nullable=True, comment="用户名")
    operation_type = Column(String(50), nullable=True, comment="操作类型")
    operation_module = Column(String(100), nullable=True, comment="操作模块")
    operation_desc = Column(Text, nullable=True, comment="操作描述")
    request_method = Column(String(10), nullable=True, comment="请求方法")
    request_url = Column(String(500), nullable=True, comment="请求URL")
    request_params = Column(Text, nullable=True, comment="请求参数")
    response_code = Column(Integer, nullable=True, comment="响应状态码")
    response_msg = Column(Text, nullable=True, comment="响应消息")
    ip_address = Column(String(50), nullable=True, comment="IP地址")
    user_agent = Column(String(500), nullable=True, comment="用户代理")
    execution_time = Column(Integer, nullable=True, comment="执行时间(ms)")
    status = Column(String(20), default="success", comment="状态")
    error_msg = Column(Text, nullable=True, comment="错误消息")
    created_at = Column(DateTime, default=datetime.utcnow, index=True, comment="创建时间")
    
    def __repr__(self):
        return f"<OperationLog(id={self.id}, operation_type='{self.operation_type}', username='{self.username}')>"
