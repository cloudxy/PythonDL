"""pytest 配置文件

此模块配置 pytest 测试环境，提供通用的 fixtures 和配置。
"""
import os
import pytest
import asyncio
import httpx
from typing import Dict, Any, AsyncGenerator, Generator


# 测试配置
TEST_BASE_URL = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "mysql+pymysql://root:root@localhost:3306/pythondl_test?charset=utf8mb4"
)


# 测试数据 fixtures
@pytest.fixture(scope="session")
def test_user_data() -> Dict[str, Any]:
    """测试用户数据"""
    return {
        "valid_user": {
            "username": "testuser_api",
            "email": "testuser@example.com",
            "password": "password123",
            "real_name": "Test User"
        },
        "invalid_user": {
            "username": "",
            "email": "invalid-email",
            "password": "",
            "real_name": ""
        }
    }


@pytest.fixture(scope="session")
def test_user_credentials() -> Dict[str, str]:
    """测试用户凭证"""
    return {
        "username": "admin",
        "password": "admin123",
        "email": "admin@example.com"
    }


@pytest.fixture(scope="session")
def stock_test_data() -> Dict[str, Any]:
    """股票测试数据"""
    return {
        "valid_stock": {
            "ts_code": "000001.SZ",
            "days": 30
        },
        "invalid_stock_code": {
            "ts_code": "INVALID.CODE",
            "days": 30
        }
    }


@pytest.fixture(scope="session")
def weather_test_data() -> Dict[str, Any]:
    """气象测试数据"""
    return {
        "valid_station": {
            "station_code": "BJ001",
            "days": 7
        },
        "invalid_station": {
            "station_code": "INVALID_STATION",
            "days": 7
        }
    }


@pytest.fixture(scope="session")
def fortune_test_data() -> Dict[str, Any]:
    """算命测试数据"""
    return {
        "feng_shui": {
            "category": "home",
            "title": "Test Feng Shui",
            "content": "Test content",
            "score": 80
        },
        "face_reading": {
            "face_part": "eye",
            "feature": "bright",
            "meaning": "Good fortune",
            "score": 85
        },
        "bazi": {
            "yearpillar": "甲子",
            "monthpillar": "乙丑",
            "daypillar": "丙寅",
            "hourpillar": "丁卯"
        },
        "fortune_telling": {
            "category": "daily",
            "prediction": "Good day ahead",
            "lucky_color": "red",
            "lucky_number": 8
        }
    }


@pytest.fixture(scope="session")
def consumption_test_data() -> Dict[str, Any]:
    """消费测试数据"""
    return {
        "gdp": {
            "region_code": "CN-BJ",
            "year": 2023,
            "gdp_value": 40000.0,
            "growth_rate": 5.2
        },
        "population": {
            "region_code": "CN-BJ",
            "year": 2023,
            "total_population": 21000000,
            "urban_population": 18000000
        },
        "economic_indicator": {
            "region_code": "CN-BJ",
            "year": 2023,
            "indicator_name": "CPI",
            "indicator_value": 2.1,
            "unit": "%"
        },
        "community": {
            "city": "北京",
            "district": "朝阳区",
            "community_name": "Test Community",
            "average_price": 80000.0
        }
    }


@pytest.fixture(scope="session")
def performance_test_config() -> Dict[str, Any]:
    """性能测试配置"""
    return {
        "concurrent_users": [1, 5, 10],
        "duration_seconds": 10,
        "response_time_threshold_ms": 3000,
        "error_threshold_percentage": 20
    }


@pytest.fixture(scope="session")
def security_test_config() -> Dict[str, Any]:
    """安全测试配置"""
    return {
        "sql_injection_payloads": [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT NULL, NULL, NULL --",
            "1' AND '1'='1",
            "admin'--",
            "1; DELETE FROM users",
            "' OR 1=1 --",
            "1' OR '1'='1' /*",
        ],
        "xss_payloads": [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'>",
            "<svg onload=alert('XSS')>",
        ]
    }


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """创建事件循环"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """创建异步 HTTP 客户端"""
    async with httpx.AsyncClient(base_url=TEST_BASE_URL) as client:
        yield client


@pytest.fixture(scope="session")
async def auth_headers(
    async_client: httpx.AsyncClient,
    test_user_credentials: Dict[str, str]
) -> AsyncGenerator[Dict[str, str], None]:
    """获取认证头部"""
    try:
        # 尝试登录获取 token
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_credentials["username"],
                "password": test_user_credentials["password"]
            }
        )
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data["access_token"]
            yield {"Authorization": f"Bearer {access_token}"}
        else:
            yield {}
    except Exception:
        yield {}


@pytest.fixture(scope="session")
async def refresh_token(
    async_client: httpx.AsyncClient,
    test_user_credentials: Dict[str, str]
) -> AsyncGenerator[str, None]:
    """获取刷新令牌"""
    try:
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_credentials["username"],
                "password": test_user_credentials["password"]
            }
        )
        
        if response.status_code == 200:
            token_data = response.json()
            yield token_data["refresh_token"]
        else:
            yield None
    except Exception:
        yield None


# 自定义 pytest 标记
def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "crud: marks tests as CRUD tests"
    )


# 命令行选项
def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--base-url",
        action="store",
        default=TEST_BASE_URL,
        help="Base URL for API tests"
    )
    parser.addoption(
        "--no-auth",
        action="store_true",
        help="Skip authentication tests"
    )


@pytest.fixture
def base_url(request) -> str:
    """获取基础 URL"""
    return request.config.getoption("--base-url")


@pytest.fixture
def skip_auth_tests(request) -> bool:
    """是否跳过认证测试"""
    return request.config.getoption("--no-auth")
