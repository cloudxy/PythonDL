"""爬虫监控指标

此模块使用 Prometheus 提供爬虫性能指标监控。
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from app.core.logger import get_logger

logger = get_logger("crawler_metrics")


@dataclass
class CrawlerMetrics:
    """爬虫指标数据类"""
    crawler_type: str = ""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_records: int = 0
    avg_response_time: float = 0.0
    error_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.successful_requests + self.failed_requests
        return self.successful_requests / max(1, total)
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)
    
    @property
    def records_per_second(self) -> float:
        """每秒记录数"""
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            return self.total_records / max(1, duration)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "crawler_type": self.crawler_type,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_records": self.total_records,
            "avg_response_time": round(self.avg_response_time, 4),
            "records_per_second": round(self.records_per_second, 2),
            "error_count": self.error_count,
        }


class PrometheusMetrics:
    """Prometheus 监控指标"""
    
    def __init__(self, crawler_type: str = "base"):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus 客户端未安装，指标监控将不可用")
            self.enabled = False
            return
        
        self.enabled = True
        self.crawler_type = crawler_type
        
        # 定义指标
        self.request_count = Counter(
            'crawler_requests_total',
            'Total number of crawler requests',
            ['crawler_type', 'status'],
            namespace='python_dl'
        )
        
        self.request_duration = Histogram(
            'crawler_request_duration_seconds',
            'Crawler request duration in seconds',
            ['crawler_type'],
            namespace='python_dl',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )
        
        self.cache_hit_rate = Gauge(
            'crawler_cache_hit_rate',
            'Crawler cache hit rate',
            ['crawler_type'],
            namespace='python_dl'
        )
        
        self.active_tasks = Gauge(
            'crawler_active_tasks',
            'Number of active crawler tasks',
            ['crawler_type'],
            namespace='python_dl'
        )
        
        self.error_count = Counter(
            'crawler_errors_total',
            'Total number of crawler errors',
            ['crawler_type', 'error_type'],
            namespace='python_dl'
        )
        
        self.records_processed = Counter(
            'crawler_records_processed_total',
            'Total number of records processed',
            ['crawler_type'],
            namespace='python_dl'
        )
        
        self.response_time = Gauge(
            'crawler_avg_response_time',
            'Average response time in seconds',
            ['crawler_type'],
            namespace='python_dl'
        )
    
    def record_request(self, status: str = "success", duration: float = 0.0):
        """记录请求"""
        if not self.enabled:
            return
        
        self.request_count.labels(
            crawler_type=self.crawler_type,
            status=status
        ).inc()
        
        if duration > 0:
            self.request_duration.labels(crawler_type=self.crawler_type).observe(duration)
    
    def record_cache_hit(self, hit_rate: float):
        """记录缓存命中"""
        if not self.enabled:
            return
        
        self.cache_hit_rate.labels(crawler_type=self.crawler_type).set(hit_rate)
    
    def record_error(self, error_type: str = "unknown"):
        """记录错误"""
        if not self.enabled:
            return
        
        self.error_count.labels(
            crawler_type=self.crawler_type,
            error_type=error_type
        ).inc()
    
    def record_records(self, count: int):
        """记录处理的数据量"""
        if not self.enabled:
            return
        
        self.records_processed.labels(crawler_type=self.crawler_type).inc(count)
    
    def record_response_time(self, avg_time: float):
        """记录平均响应时间"""
        if not self.enabled:
            return
        
        self.response_time.labels(crawler_type=self.crawler_type).set(avg_time)
    
    def set_active_tasks(self, count: int):
        """设置活跃任务数"""
        if not self.enabled:
            return
        
        self.active_tasks.labels(crawler_type=self.crawler_type).set(count)
    
    def get_metrics(self) -> Optional[bytes]:
        """获取 Prometheus 格式指标"""
        if not self.enabled:
            return None
        
        return generate_latest()


class MetricsCollector:
    """指标收集器（不依赖 Prometheus）"""
    
    def __init__(self, crawler_type: str = "base"):
        self.crawler_type = crawler_type
        self.metrics = CrawlerMetrics(crawler_type=crawler_type)
        self.response_times: list = []
        self.start_time = datetime.now()
    
    def record_request_start(self):
        """记录请求开始"""
        self.metrics.total_requests += 1
        self._request_start = time.time()
    
    def record_request_end(self, success: bool = True, records_count: int = 0):
        """记录请求结束"""
        duration = time.time() - self._request_start
        self.response_times.append(duration)
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            self.metrics.error_count += 1
        
        self.metrics.total_records += records_count
        self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def record_cache(self, hit: bool):
        """记录缓存"""
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
    
    def get_metrics(self) -> CrawlerMetrics:
        """获取指标"""
        self.metrics.end_time = datetime.now()
        return self.metrics
    
    def reset(self):
        """重置指标"""
        self.metrics = CrawlerMetrics(crawler_type=self.crawler_type)
        self.response_times = []
        self.start_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.get_metrics().to_dict()


# 装饰器：记录指标
def track_metrics(metrics_collector: MetricsCollector):
    """指标追踪装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics_collector.record_request_start()
            try:
                result = await func(*args, **kwargs)
                records_count = len(result) if isinstance(result, (list, dict)) else 0
                metrics_collector.record_request_end(success=True, records_count=records_count)
                return result
            except Exception as e:
                metrics_collector.record_request_end(success=False)
                raise
        return wrapper
    return decorator


# 全局指标收集器
_metrics_collectors: Dict[str, MetricsCollector] = {}


def get_metrics_collector(crawler_type: str) -> MetricsCollector:
    """获取指标收集器"""
    if crawler_type not in _metrics_collectors:
        _metrics_collectors[crawler_type] = MetricsCollector(crawler_type)
    return _metrics_collectors[crawler_type]


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """获取所有爬虫的指标"""
    return {
        crawler_type: collector.to_dict()
        for crawler_type, collector in _metrics_collectors.items()
    }
