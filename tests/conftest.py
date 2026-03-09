"""自动化测试配置

此模块用于配置自动化测试环境
"""
import pytest
from playwright.sync_api import Page, BrowserContext
import os

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
TEST_USER = {
    "username": os.getenv("TEST_USERNAME", "admin"),
    "password": os.getenv("TEST_PASSWORD", "admin123")
}


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """配置浏览器上下文参数"""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "locale": "zh-CN",
        "timezone_id": "Asia/Shanghai",
    }


@pytest.fixture
def authenticated_page(page: Page) -> Page:
    """已认证的页面fixture
    
    此fixture会自动登录并返回已认证的页面
    """
    def login():
        page.goto(f"{BASE_URL}/static/pages/auth/login.html")
        
        page.fill("input[name='username']", TEST_USER["username"])
        page.fill("input[name='password']", TEST_USER["password"])
        page.click("button[type='submit']")
        
        page.wait_for_url(f"{BASE_URL}/static/pages/dashboard.html", timeout=5000)
    
    login()
    return page


@pytest.fixture
def base_url() -> str:
    """基础URL fixture"""
    return BASE_URL


@pytest.fixture
def test_user() -> dict:
    """测试用户fixture"""
    return TEST_USER
