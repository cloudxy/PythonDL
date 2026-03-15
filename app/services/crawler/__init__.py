"""爬虫模块

提供统一的异步爬虫框架，包含：
- AI 驱动的智能爬虫 (基于 LLM)
- 代理池管理
- 监控指标
- 智能配置系统
"""
from app.services.crawler.base_crawler import (
    AsyncBaseCrawler,
    RequestConfig,
    CrawlerStatus,
    CrawlerCircuitBreaker,
    CircuitBreakerError,
    DataPipeline,
    DataDeduplicator,
)

from app.services.crawler.proxy_pool import (
    ProxyPool,
    Proxy,
    ProxyStatus,
    ProxyMiddleware,
)

from app.services.crawler.crawler_metrics import (
    PrometheusMetrics,
    MetricsCollector,
    CrawlerMetrics,
    get_metrics_collector,
    get_all_metrics,
)

# AI 爬虫和配置
from app.services.crawler.ai_crawler import (
    AIBaseCrawler,
    SmartScraperCrawler,
    SearchGraphCrawler,
    CrawlerGraphConfig,
    LLMConfig,
    LLMProvider,
    CrawlerExecutionState,
)

from app.services.crawler.llm_service import (
    BaseLLMService,
    OllamaService,
    OpenAIService,
    GroqService,
    LLMServiceFactory,
    LLMProviderType,
)

from app.services.crawler.crawler_config import (
    CrawlerConfigManager,
    CrawlerProfile,
    DataSourceConfig,
    ExtractionPrompt,
    CrawlerType,
    get_config_manager,
    get_default_profile,
)

__all__ = [
    # 基类
    'AsyncBaseCrawler',
    'RequestConfig',
    'CrawlerStatus',
    'CrawlerCircuitBreaker',
    'CircuitBreakerError',
    'DataPipeline',
    'DataDeduplicator',
    
    # 代理池
    'ProxyPool',
    'Proxy',
    'ProxyStatus',
    'ProxyMiddleware',
    
    # 监控指标
    'PrometheusMetrics',
    'MetricsCollector',
    'CrawlerMetrics',
    'get_metrics_collector',
    'get_all_metrics',
    
    # AI 爬虫
    'AIBaseCrawler',
    'SmartScraperCrawler',
    'SearchGraphCrawler',
    'CrawlerGraphConfig',
    'LLMConfig',
    'LLMProvider',
    'CrawlerExecutionState',
    
    # LLM 服务
    'BaseLLMService',
    'OllamaService',
    'OpenAIService',
    'GroqService',
    'LLMServiceFactory',
    'LLMProviderType',
    
    # 配置管理
    'CrawlerConfigManager',
    'CrawlerProfile',
    'DataSourceConfig',
    'ExtractionPrompt',
    'CrawlerType',
    'get_config_manager',
    'get_default_profile',
]
