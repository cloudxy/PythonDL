"""爬虫详细配置管理 - 支持认证、API、Token 等场景

此模块提供爬虫的完整配置系统，包括：
- 网站登录认证（用户名密码、Cookie）
- API 接口认证（API Key、Access Token）
- OAuth2 认证
- 请求头配置
- 代理配置
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from app.core.logger import get_logger

logger = get_logger("crawler_config")


class CrawlerType(str, Enum):
    """爬虫类型"""
    STOCK = "stock"
    WEATHER = "weather"
    CONSUMPTION = "consumption"
    FORTUNE = "fortune"
    GENERAL = "general"
    FINANCE = "finance"  # 金融数据采集
    FINANCE_AKSHARE = "finance_akshare"  # 基于 AKShare 的金融数据
    FINANCE_AI = "finance_ai"  # 基于 AI 的金融数据


class AuthType(str, Enum):
    """认证类型"""
    NONE = "none"
    LOGIN = "login"  # 用户名密码登录
    API_KEY = "api_key"  # API Key 认证
    TOKEN = "token"  # Access Token
    OAUTH2 = "oauth2"  # OAuth2
    COOKIE = "cookie"  # Cookie 认证
    BASIC = "basic"  # Basic Auth
    BEARER = "bearer"  # Bearer Token


@dataclass
class LoginConfig:
    """登录配置"""
    login_url: str = ""
    username: str = ""
    password: str = ""
    login_method: str = "POST"  # POST or GET
    form_data: Dict[str, str] = field(default_factory=dict)
    success_indicator: str = ""  # 登录成功标识
    cookie_name: str = ""
    save_cookie: bool = True
    auto_refresh: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "login_url": self.login_url,
            "username": self.username,
            "password": "***" if self.password else None,
            "login_method": self.login_method,
            "form_data": self.form_data,
            "success_indicator": self.success_indicator,
            "cookie_name": self.cookie_name,
            "save_cookie": self.save_cookie,
            "auto_refresh": self.auto_refresh,
        }


@dataclass
class APIConfig:
    """API 接口配置"""
    base_url: str = ""
    api_version: str = "v1"
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    token_url: str = ""
    token_type: str = "Bearer"  # Bearer, Basic, etc.
    grant_type: str = "client_credentials"  # OAuth2 grant type
    client_id: str = ""
    client_secret: str = ""
    scope: str = ""
    refresh_token: str = ""
    token_expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_version": self.api_version,
            "api_key": self.api_key[:8] + "..." if self.api_key else None,
            "has_secret": bool(self.api_secret),
            "has_token": bool(self.access_token),
            "token_type": self.token_type,
            "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
        }
    
    def is_token_expired(self) -> bool:
        """检查 token 是否过期"""
        if not self.token_expires_at:
            return False
        return datetime.now() > self.token_expires_at


@dataclass
class HeadersConfig:
    """请求头配置"""
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    accept: str = "application/json, text/html, */*"
    accept_language: str = "zh-CN,zh;q=0.9,en;q=0.8"
    accept_encoding: str = "gzip, deflate, br"
    content_type: str = "application/json"
    referer: str = ""
    origin: str = ""
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": self.accept,
            "Accept-Language": self.accept_language,
            "Accept-Encoding": self.accept_encoding,
            "Content-Type": self.content_type,
        }
        if self.referer:
            headers["Referer"] = self.referer
        if self.origin:
            headers["Origin"] = self.origin
        headers.update(self.custom_headers)
        return headers


@dataclass
class ProxyConfig:
    """代理配置"""
    enabled: bool = False
    proxy_url: str = ""
    proxy_username: str = ""
    proxy_password: str = ""
    proxy_pool: List[str] = field(default_factory=list)
    rotation_strategy: str = "random"  # random, round_robin, sticky
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "has_proxy": bool(self.proxy_url),
            "has_pool": bool(self.proxy_pool),
            "rotation_strategy": self.rotation_strategy,
            "max_retries": self.max_retries,
        }


@dataclass
class RateLimitConfig:
    """频率限制配置"""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    requests_per_day: int = 86400
    burst_size: int = 10
    concurrent_limit: int = 5
    delay_between_requests: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "burst_size": self.burst_size,
            "concurrent_limit": self.concurrent_limit,
            "delay_between_requests": self.delay_between_requests,
        }


@dataclass
class ScheduleConfig:
    """调度配置"""
    enabled: bool = True
    cron_expression: str = "0 * * * *"
    interval_seconds: int = 3600
    execute_at: List[str] = field(default_factory=list)
    timezone: str = "Asia/Shanghai"
    retry_on_failure: bool = True
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "execute_at": self.execute_at,
            "timezone": self.timezone,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
        }


@dataclass
class DatabaseFieldMapping:
    """数据库字段映射配置"""
    db_field: str
    column_type: str
    source_field: str
    required: bool = False
    nullable: bool = True
    default_value: Any = None
    transform: Optional[str] = None
    description: str = ""
    comment: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_field": self.db_field,
            "column_type": self.column_type,
            "source_field": self.source_field,
            "required": self.required,
            "nullable": self.nullable,
            "default_value": self.default_value,
            "transform": self.transform,
            "description": self.description,
            "comment": self.comment,
        }


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    url: str
    method: str = "GET"
    headers: Optional[HeadersConfig] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retries: int = 3
    cache_ttl: int = 3600
    enabled: bool = True
    auth_type: AuthType = AuthType.NONE
    login_config: Optional[LoginConfig] = None
    api_config: Optional[APIConfig] = None
    proxy_config: Optional[ProxyConfig] = None
    rate_limit: Optional[RateLimitConfig] = None
    db_fields: List[DatabaseFieldMapping] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "method": self.method,
            "headers": self.headers.to_dict() if self.headers else None,
            "params": self.params,
            "timeout": self.timeout,
            "retries": self.retries,
            "cache_ttl": self.cache_ttl,
            "enabled": self.enabled,
            "auth_type": self.auth_type.value,
            "login_config": self.login_config.to_dict() if self.login_config else None,
            "api_config": self.api_config.to_dict() if self.api_config else None,
            "proxy_config": self.proxy_config.to_dict() if self.proxy_config else None,
            "rate_limit": self.rate_limit.to_dict() if self.rate_limit else None,
            "db_fields": [f.to_dict() for f in self.db_fields],
            "description": self.description,
        }


@dataclass
class TableMapping:
    """数据库表映射配置"""
    table_name: str
    model_class: str
    primary_key: str = "id"
    unique_keys: List[str] = field(default_factory=list)
    index_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "model_class": self.model_class,
            "primary_key": self.primary_key,
            "unique_keys": self.unique_keys,
            "index_fields": self.index_fields,
        }


@dataclass
class DataPipelineStep:
    """数据管道步骤"""
    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "enabled": self.enabled,
        }


@dataclass
class ExtractionPrompt:
    """提取 Prompt"""
    name: str
    description: str
    prompt: str
    output_format: str = "json"
    db_fields: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "output_format": self.output_format,
            "db_fields": self.db_fields,
            "examples": self.examples,
        }


@dataclass
class CrawlerProfile:
    """爬虫配置档案"""
    crawler_type: CrawlerType
    name: str
    description: str
    version: str = "1.0.0"
    table_mapping: Optional[TableMapping] = None
    default_data_sources: List[DataSourceConfig] = field(default_factory=list)
    default_prompt: Optional[ExtractionPrompt] = None
    llm_model: str = "llama3.2"
    rate_limit: Optional[RateLimitConfig] = None
    schedule: Optional[ScheduleConfig] = None
    data_pipeline: List[DataPipelineStep] = field(default_factory=list)
    cache_enabled: bool = True
    dedup_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "crawler_type": self.crawler_type.value,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "table_mapping": self.table_mapping.to_dict() if self.table_mapping else None,
            "default_data_sources": [ds.to_dict() for ds in self.default_data_sources],
            "default_prompt": self.default_prompt.to_dict() if self.default_prompt else None,
            "llm_model": self.llm_model,
            "rate_limit": self.rate_limit.to_dict() if self.rate_limit else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "data_pipeline": [step.to_dict() for step in self.data_pipeline],
            "cache_enabled": self.cache_enabled,
            "dedup_enabled": self.dedup_enabled,
        }


class CrawlerConfigManager:
    """爬虫配置管理器"""
    
    def __init__(self):
        self.profiles: Dict[CrawlerType, CrawlerProfile] = {}
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self):
        """初始化默认配置档案"""
        
        # ========== 股票爬虫配置 ==========
        stock_profile = CrawlerProfile(
            crawler_type=CrawlerType.STOCK,
            name="股票数据采集",
            description="采集 A 股、港股、美股等股票行情数据",
            version="1.0.0",
            table_mapping=TableMapping(
                table_name="stocks",
                model_class="Stock",
                unique_keys=["ts_code", "trade_date"]
            ),
            llm_model="llama3.2",
            rate_limit=RateLimitConfig(
                requests_per_second=0.5,
                requests_per_minute=30,
                concurrent_limit=3
            ),
            schedule=ScheduleConfig(
                cron_expression="*/5 9-15 * * 1-5",
                interval_seconds=300,
                execute_at=["09:30", "11:30", "13:00", "15:00"]
            ),
            default_data_sources=[
                # 数据源 1：新浪财经 - 无需认证
                DataSourceConfig(
                    name="新浪财经 - 实时行情",
                    url="http://hq.sinajs.cn/list={symbol}",
                    method="GET",
                    headers=HeadersConfig(
                        referer="https://finance.sina.com.cn/",
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    ),
                    timeout=10,
                    retries=3,
                    cache_ttl=60,
                    auth_type=AuthType.NONE,
                    description="A 股实时行情，无需认证",
                    db_fields=[
                        DatabaseFieldMapping("ts_code", "String(20)", "$.symbol", required=True),
                        DatabaseFieldMapping("trade_date", "Date", "$.date", required=True),
                        DatabaseFieldMapping("open", "Numeric(10,3)", "$.open"),
                        DatabaseFieldMapping("high", "Numeric(10,3)", "$.high"),
                        DatabaseFieldMapping("low", "Numeric(10,3)", "$.low"),
                        DatabaseFieldMapping("close", "Numeric(10,3)", "$.price"),
                        DatabaseFieldMapping("volume", "BigInteger", "$.volume"),
                        DatabaseFieldMapping("amount", "Numeric(20,3)", "$.amount"),
                    ]
                ),
                # 数据源 2：聚宽数据 - 需要 API Key
                DataSourceConfig(
                    name="聚宽数据 API",
                    url="https://api.joinquant.com/data/stocks",
                    method="GET",
                    auth_type=AuthType.API_KEY,
                    api_config=APIConfig(
                        base_url="https://api.joinquant.com",
                        api_version="v1",
                        api_key="YOUR_JQ_API_KEY",
                        token_type="Bearer"
                    ),
                    headers=HeadersConfig(
                        custom_headers={"X-API-Key": "{api_key}"}
                    ),
                    timeout=15,
                    cache_ttl=300,
                    description="聚宽股票数据，需要 API Key 认证",
                    db_fields=[
                        DatabaseFieldMapping("ts_code", "String(20)", "$.code", required=True),
                        DatabaseFieldMapping("close", "Numeric(10,3)", "$.close"),
                    ]
                ),
            ],
            default_prompt=ExtractionPrompt(
                name="股票数据提取",
                description="从网页中提取股票行情数据",
                prompt="请从内容中提取股票数据：ts_code(股票代码)、trade_date(交易日期)、open(开盘价)、high(最高价)、low(最低价)、close(收盘价)、volume(成交量)、amount(成交额)。以 JSON 格式返回。",
                output_format="json",
                db_fields=["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount"]
            ),
            data_pipeline=[
                DataPipelineStep("数据清洗", "transform", {"remove_null": True}),
                DataPipelineStep("数据验证", "validate", {"required_fields": ["ts_code", "trade_date"]}),
                DataPipelineStep("数据去重", "filter", {"key": ["ts_code", "trade_date"]}),
                DataPipelineStep("数据入库", "save", {"table": "stocks", "model": "Stock"}),
            ]
        )
        self.profiles[CrawlerType.STOCK] = stock_profile
        
        # ========== 气象爬虫配置 ==========
        weather_profile = CrawlerProfile(
            crawler_type=CrawlerType.WEATHER,
            name="气象数据采集",
            description="采集全国气象数据",
            version="1.0.0",
            table_mapping=TableMapping(
                table_name="weather_data",
                model_class="Weather",
                unique_keys=["station_id", "record_date"]
            ),
            llm_model="llama3.2",
            rate_limit=RateLimitConfig(
                requests_per_second=1.0,
                requests_per_minute=60,
                concurrent_limit=5
            ),
            schedule=ScheduleConfig(
                cron_expression="0 */6 * * *",
                interval_seconds=21600,
                execute_at=["00:00", "06:00", "12:00", "18:00"]
            ),
            default_data_sources=[
                # 数据源 1：和风天气 - 需要 API Key
                DataSourceConfig(
                    name="和风天气 API",
                    url="https://devapi.qweather.com/v7/weather/now",
                    method="GET",
                    auth_type=AuthType.API_KEY,
                    api_config=APIConfig(
                        base_url="https://devapi.qweather.com",
                        api_version="v7",
                        api_key="YOUR_QWEATHER_KEY",
                        token_type="Bearer"
                    ),
                    params={"key": "{api_key}", "location": "{location_id}"},
                    timeout=10,
                    cache_ttl=1800,
                    description="和风天气实时数据，需要 API Key",
                    db_fields=[
                        DatabaseFieldMapping("station_id", "String(20)", "$.fxLink", required=True),
                        DatabaseFieldMapping("record_date", "Date", "$.updateTime", required=True),
                        DatabaseFieldMapping("max_temp", "Numeric(5,2)", "$.now.temp"),
                        DatabaseFieldMapping("avg_humidity", "Numeric(5,2)", "$.now.humidity"),
                        DatabaseFieldMapping("weather_type", "String(50)", "$.now.text"),
                        DatabaseFieldMapping("wind_direction", "String(20)", "$.now.windDir"),
                        DatabaseFieldMapping("wind_speed", "Numeric(5,2)", "$.now.windScale"),
                    ]
                ),
                # 数据源 2：中国天气网 - 无需认证
                DataSourceConfig(
                    name="中国天气网",
                    url="http://www.weather.com.cn/weather/{city_code}.shtml",
                    method="GET",
                    auth_type=AuthType.NONE,
                    timeout=15,
                    cache_ttl=1800,
                    description="城市天气数据，无需认证",
                    db_fields=[
                        DatabaseFieldMapping("city", "String(50)", "$.city", required=True),
                        DatabaseFieldMapping("weather", "String(50)", "$.weather"),
                        DatabaseFieldMapping("temperature", "String(20)", "$.temp"),
                    ]
                ),
            ],
            default_prompt=ExtractionPrompt(
                name="气象数据提取",
                description="从网页中提取气象数据",
                prompt="请从内容中提取气象数据：station_id(站点 ID)、record_date(记录日期)、max_temp(最高温度)、min_temp(最低温度)、avg_humidity(平均湿度)、weather_type(天气类型)、wind_direction(风向)、wind_speed(风速)。以 JSON 格式返回。",
                output_format="json",
                db_fields=["station_id", "record_date", "max_temp", "min_temp", "avg_humidity", "weather_type", "wind_direction", "wind_speed"]
            ),
            data_pipeline=[
                DataPipelineStep("温度单位转换", "transform", {"celsius": True}),
                DataPipelineStep("数据验证", "validate", {"required_fields": ["station_id", "record_date"]}),
                DataPipelineStep("数据入库", "save", {"table": "weather_data", "model": "Weather"}),
            ]
        )
        self.profiles[CrawlerType.WEATHER] = weather_profile
        
        # ========== 消费数据爬虫配置 ==========
        consumption_profile = CrawlerProfile(
            crawler_type=CrawlerType.CONSUMPTION,
            name="宏观消费数据采集",
            description="采集 GDP、CPI、PPI、PMI 等宏观经济数据",
            version="1.0.0",
            table_mapping=TableMapping(
                table_name="gdp_data",
                model_class="GDPData",
                unique_keys=["region_code", "year", "quarter"]
            ),
            llm_model="llama3.2",
            rate_limit=RateLimitConfig(
                requests_per_second=0.2,
                requests_per_minute=12,
                concurrent_limit=2
            ),
            schedule=ScheduleConfig(
                cron_expression="0 9 1 * *",
                interval_seconds=86400,
                execute_at=["09:00"]
            ),
            default_data_sources=[
                # 数据源 1：国家统计局 - 无需认证
                DataSourceConfig(
                    name="国家统计局",
                    url="https://data.stats.gov.cn/easyquery.htm",
                    method="GET",
                    auth_type=AuthType.NONE,
                    timeout=30,
                    cache_ttl=86400,
                    description="国家宏观经济数据，无需认证",
                    db_fields=[
                        DatabaseFieldMapping("region_code", "String(20)", "$.region_code", required=True),
                        DatabaseFieldMapping("region_name", "String(100)", "$.region_name", required=True),
                        DatabaseFieldMapping("year", "Integer", "$.year", required=True),
                        DatabaseFieldMapping("gdp", "BigInteger", "$.gdp"),
                        DatabaseFieldMapping("gdp_growth", "Numeric(10,3)", "$.growth_rate"),
                    ]
                ),
            ],
            default_prompt=ExtractionPrompt(
                name="宏观经济数据提取",
                description="从网页中提取宏观经济数据",
                prompt="请从内容中提取宏观经济数据：region_code(区域代码)、region_name(区域名称)、year(年份)、gdp(GDP 万元)、gdp_growth(GDP 增长率)。以 JSON 格式返回。",
                output_format="json",
                db_fields=["region_code", "region_name", "year", "gdp", "gdp_growth"]
            ),
            data_pipeline=[
                DataPipelineStep("数据格式化", "transform", {"date_format": "YYYY-MM"}),
                DataPipelineStep("数据验证", "validate", {"required_fields": ["region_code", "year"]}),
                DataPipelineStep("数据入库", "save", {"table": "gdp_data", "model": "GDPData"}),
            ]
        )
        self.profiles[CrawlerType.CONSUMPTION] = consumption_profile
        
        # ========== 算命数据爬虫配置 ==========
        fortune_profile = CrawlerProfile(
            crawler_type=CrawlerType.FORTUNE,
            name="周易算命数据采集",
            description="采集周易、八字、面相、风水等传统文化数据",
            version="1.0.0",
            table_mapping=TableMapping(
                table_name="zhou_yi",
                model_class="ZhouYi",
                unique_keys=["gua_name"]
            ),
            llm_model="llama3.2",
            rate_limit=RateLimitConfig(
                requests_per_second=0.1,
                requests_per_minute=6,
                concurrent_limit=1
            ),
            schedule=ScheduleConfig(
                cron_expression="0 2 * * *",
                interval_seconds=86400,
                execute_at=["02:00"]
            ),
            default_data_sources=[
                # 数据源 1：百度百科 - 无需认证
                DataSourceConfig(
                    name="百度百科 - 周易",
                    url="https://baike.baidu.com/item/{keyword}",
                    method="GET",
                    auth_type=AuthType.NONE,
                    headers=HeadersConfig(
                        referer="https://baike.baidu.com/",
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    ),
                    timeout=15,
                    cache_ttl=86400,
                    description="周易百科知识，无需认证",
                    db_fields=[
                        DatabaseFieldMapping("gua_name", "String(50)", "$.title", required=True),
                        DatabaseFieldMapping("gua_symbol", "String(20)", "$.symbol"),
                        DatabaseFieldMapping("gua_ci", "Text", "$.content"),
                        DatabaseFieldMapping("explanation", "Text", "$.explanation"),
                        DatabaseFieldMapping("source", "String(200)", "$._url"),
                    ]
                ),
            ],
            default_prompt=ExtractionPrompt(
                name="周易数据提取",
                description="从网页中提取周易相关数据",
                prompt="请从内容中提取周易相关数据：gua_name(卦名)、gua_symbol(卦象)、gua_ci(卦辞)、explanation(解释)、source(出处)。以 JSON 格式返回。",
                output_format="json",
                db_fields=["gua_name", "gua_symbol", "gua_ci", "explanation", "source"]
            ),
            data_pipeline=[
                DataPipelineStep("文本清洗", "transform", {"remove_html": True}),
                DataPipelineStep("数据验证", "validate", {"required_fields": ["gua_name"]}),
                DataPipelineStep("数据入库", "save", {"table": "zhou_yi", "model": "ZhouYi"}),
            ]
        )
        self.profiles[CrawlerType.FORTUNE] = fortune_profile
        
        # ========== 通用爬虫配置 ==========
        general_profile = CrawlerProfile(
            crawler_type=CrawlerType.GENERAL,
            name="通用网页采集",
            description="通用的网页数据采集，支持自定义配置",
            version="1.0.0",
            table_mapping=None,
            llm_model="llama3.2",
            rate_limit=RateLimitConfig(
                requests_per_second=1.0,
                requests_per_minute=60,
                concurrent_limit=5
            ),
            schedule=ScheduleConfig(
                enabled=False,
                interval_seconds=3600
            ),
            default_data_sources=[],
            default_prompt=ExtractionPrompt(
                name="通用数据提取",
                description="从网页中提取用户指定的数据",
                prompt="请从内容中提取我需要的信息。{custom_requirements}。以 JSON 格式返回。",
                output_format="json",
                db_fields=[]
            ),
            data_pipeline=[
                DataPipelineStep("数据清洗", "transform", {}),
                DataPipelineStep("数据入库", "save", {}),
            ]
        )
        self.profiles[CrawlerType.GENERAL] = general_profile
    
    def get_profile(self, crawler_type: CrawlerType) -> CrawlerProfile:
        """获取爬虫配置档案"""
        return self.profiles.get(crawler_type)
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """获取所有配置档案"""
        return {
            crawler_type.value: profile.to_dict()
            for crawler_type, profile in self.profiles.items()
        }
    
    def get_profile_detail(self, crawler_type: str) -> Dict[str, Any]:
        """获取爬虫配置详情"""
        profile = self.profiles.get(CrawlerType(crawler_type))
        if not profile:
            return {}
        
        result = profile.to_dict()
        
        # 添加表结构信息
        if profile.table_mapping:
            result["table_structure"] = {
                "table_name": profile.table_mapping.table_name,
                "model_class": profile.table_mapping.model_class,
                "fields": []
            }
            
            for ds in profile.default_data_sources:
                for field in ds.db_fields:
                    result["table_structure"]["fields"].append(field.to_dict())
        
        return result


# 全局配置管理器实例
_config_manager: Optional[CrawlerConfigManager] = None


def get_config_manager() -> CrawlerConfigManager:
    """获取配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = CrawlerConfigManager()
    return _config_manager


def get_default_profile(crawler_type: str) -> Dict[str, Any]:
    """获取默认爬虫配置"""
    manager = get_config_manager()
    profile = manager.get_profile(CrawlerType(crawler_type))
    return profile.to_dict() if profile else {}


def get_profile_detail(crawler_type: str) -> Dict[str, Any]:
    """获取爬虫配置详情"""
    manager = get_config_manager()
    return manager.get_profile_detail(crawler_type)


# ========== 金融数据专用配置 ==========

@dataclass
class StockFinanceConfig:
    """股票金融数据配置"""
    data_type: str = "realtime_quote"  # realtime_quote/history/financial/stock_basic
    market: str = "A 股"  # A 股/港股/美股
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: str = "daily"  # daily/weekly/monthly
    adjust: str = "qfq"  # 复权类型：qfq/hfq/no
    fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_type": self.data_type,
            "market": self.market,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "period": self.period,
            "adjust": self.adjust,
            "fields": self.fields,
        }


@dataclass
class FinancialIndicatorConfig:
    """财务指标配置"""
    ts_code: str = ""
    start_year: int = 2020
    end_year: int = 2024
    indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "indicators": self.indicators,
        }


@dataclass
class FinanceDataSourceConfig:
    """金融数据源配置"""
    name: str
    source_type: str  # akshare/web/api
    url: str = ""
    method: str = "GET"
    config: Dict[str, Any] = field(default_factory=dict)
    headers: Optional[HeadersConfig] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retries: int = 3
    cache_ttl: int = 3600
    enabled: bool = True
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "url": self.url,
            "method": self.method,
            "config": self.config,
            "headers": self.headers.to_dict() if self.headers else None,
            "params": self.params,
            "timeout": self.timeout,
            "retries": self.retries,
            "cache_ttl": self.cache_ttl,
            "enabled": self.enabled,
            "description": self.description,
        }


@dataclass
class FinanceCrawlerConfig:
    """金融爬虫配置"""
    crawler_type: str = "finance"
    name: str = "金融数据采集"
    description: str = ""
    data_sources: List[FinanceDataSourceConfig] = field(default_factory=list)
    stock_config: Optional[StockFinanceConfig] = None
    indicator_config: Optional[FinancialIndicatorConfig] = None
    llm_config: Optional[Dict[str, Any]] = None
    rate_limit: Optional[RateLimitConfig] = None
    schedule: Optional[ScheduleConfig] = None
    data_pipeline: List[DataPipelineStep] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "crawler_type": self.crawler_type,
            "name": self.name,
            "description": self.description,
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "stock_config": self.stock_config.to_dict() if self.stock_config else None,
            "indicator_config": self.indicator_config.to_dict() if self.indicator_config else None,
            "llm_config": self.llm_config,
            "rate_limit": self.rate_limit.to_dict() if self.rate_limit else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "data_pipeline": [step.to_dict() for step in self.data_pipeline],
        }
