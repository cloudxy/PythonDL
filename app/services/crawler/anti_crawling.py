"""高级防爬模块

此模块提供爬虫的反反爬虫能力，包括 User-Agent 池、浏览器指纹伪装、请求指纹等功能。
"""
import asyncio
import hashlib
import json
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from app.core.logger import get_logger

logger = get_logger("anti_crawling")


@dataclass
class UserAgent:
    """User-Agent 数据类"""
    ua: str
    browser: str = ""
    os: str = ""
    device: str = "desktop"
    success_rate: float = 1.0
    use_count: int = 0
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ua": self.ua,
            "browser": self.browser,
            "os": self.os,
            "device": self.device,
            "success_rate": self.success_rate,
            "use_count": self.use_count
        }


class UserAgentPool:
    """User-Agent 池"""
    
    # 常见浏览器 User-Agent
    CHROME_UAS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    ]
    
    FIREFOX_UAS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    ]
    
    SAFARI_UAS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    ]
    
    EDGE_UAS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]
    
    MOBILE_UAS = [
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    ]
    
    def __init__(self):
        self.pool: List[UserAgent] = []
        self.initialize()
    
    def initialize(self):
        """初始化 UA 池"""
        # 添加 Chrome UA
        for ua in self.CHROME_UAS:
            self.pool.append(UserAgent(
                ua=ua,
                browser="Chrome",
                os="Windows" if "Windows" in ua else "Mac" if "Mac" in ua else "Linux",
                device="desktop"
            ))
        
        # 添加 Firefox UA
        for ua in self.FIREFOX_UAS:
            self.pool.append(UserAgent(
                ua=ua,
                browser="Firefox",
                os="Windows" if "Windows" in ua else "Mac" if "Mac" in ua else "Linux",
                device="desktop"
            ))
        
        # 添加 Safari UA
        for ua in self.SAFARI_UAS:
            self.pool.append(UserAgent(
                ua=ua,
                browser="Safari",
                os="Mac" if "Mac" in ua else "iOS",
                device="mobile" if "Mobile" in ua or "iPhone" in ua else "desktop"
            ))
        
        # 添加 Edge UA
        for ua in self.EDGE_UAS:
            self.pool.append(UserAgent(
                ua=ua,
                browser="Edge",
                os="Windows" if "Windows" in ua else "Mac",
                device="desktop"
            ))
        
        # 添加移动 UA
        for ua in self.MOBILE_UAS:
            self.pool.append(UserAgent(
                ua=ua,
                browser="Chrome" if "Chrome" in ua else "Safari",
                os="Android" if "Android" in ua else "iOS",
                device="mobile"
            ))
        
        logger.info(f"初始化 User-Agent 池，共 {len(self.pool)} 个 UA")
    
    def get_random(self, device_type: Optional[str] = None) -> str:
        """随机获取一个 User-Agent"""
        if device_type:
            available = [ua for ua in self.pool if ua.device == device_type]
            if available:
                ua = random.choice(available)
                ua.use_count += 1
                ua.last_used = datetime.now()
                return ua.ua
        
        ua = random.choice(self.pool)
        ua.use_count += 1
        ua.last_used = datetime.now()
        return ua.ua
    
    def get_best(self) -> str:
        """获取成功率最高的 User-Agent"""
        if not self.pool:
            return self.get_random()
        
        # 按成功率和使用次数排序
        sorted_pool = sorted(
            self.pool,
            key=lambda x: (x.success_rate, -x.use_count),
            reverse=True
        )
        
        # 从前 20% 中随机选择
        top_count = max(1, len(sorted_pool) // 5)
        ua = random.choice(sorted_pool[:top_count])
        ua.use_count += 1
        ua.last_used = datetime.now()
        return ua.ua
    
    def record_success(self, ua: str):
        """记录成功的 UA"""
        for user_agent in self.pool:
            if user_agent.ua == ua:
                user_agent.success_rate = min(1.0, user_agent.success_rate + 0.05)
                break
    
    def record_failure(self, ua: str):
        """记录失败的 UA"""
        for user_agent in self.pool:
            if user_agent.ua == ua:
                user_agent.success_rate = max(0.0, user_agent.success_rate - 0.1)
                break


class BrowserFingerprint:
    """浏览器指纹生成器"""
    
    @staticmethod
    def generate_screen_resolution() -> str:
        """生成屏幕分辨率"""
        resolutions = [
            "1920x1080", "1366x768", "2560x1440", "1440x900",
            "1536x864", "1600x900", "1280x720", "3840x2160"
        ]
        return random.choice(resolutions)
    
    @staticmethod
    def generate_timezone() -> str:
        """生成时区"""
        timezones = [
            "Asia/Shanghai", "Asia/Chongqing", "Asia/Urumqi",
            "UTC+8", "Asia/Hong_Kong", "Asia/Taipei"
        ]
        return random.choice(timezones)
    
    @staticmethod
    def generate_language() -> str:
        """生成语言设置"""
        languages = [
            "zh-CN,zh;q=0.9,en;q=0.8",
            "zh-CN,zh;q=0.9",
            "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7",
            "en-US,en;q=0.9,zh-CN;q=0.8",
        ]
        return random.choice(languages)
    
    @staticmethod
    def generate_platform() -> str:
        """生成平台"""
        platforms = [
            "Win32", "Win64", "MacIntel", "Linux x86_64"
        ]
        return random.choice(platforms)
    
    @staticmethod
    def generate_webgl_vendor() -> str:
        """生成 WebGL 厂商"""
        vendors = [
            "Intel Inc.", "NVIDIA Corporation", "AMD",
            "Apple", "Google Inc.", "Microsoft"
        ]
        return random.choice(vendors)
    
    @staticmethod
    def generate_webgl_renderer() -> str:
        """生成 WebGL 渲染器"""
        renderers = [
            "Intel Iris OpenGL Engine",
            "NVIDIA GeForce GTX 1080 OpenGL Engine",
            "AMD Radeon Pro 560 OpenGL Engine",
            "Apple M1",
            "ANGLE (Intel, Intel(R) HD Graphics 630",
        ]
        return random.choice(renderers)
    
    @staticmethod
    def generate_canvas_fingerprint() -> str:
        """生成 Canvas 指纹"""
        # 模拟 Canvas 指纹哈希
        data = f"{random.random()}{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    @staticmethod
    def generate_audio_fingerprint() -> str:
        """生成 Audio 指纹"""
        data = f"{random.random()}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    @staticmethod
    def generate_fonts() -> List[str]:
        """生成字体列表"""
        common_fonts = [
            "Arial", "Helvetica", "Times New Roman", "Courier New",
            "Verdana", "Georgia", "Palatino", "Garamond", "Bookman",
            "Comic Sans MS", "Trebuchet MS", "Arial Black", "Impact",
            "微软雅黑", "宋体", "黑体", "楷体", "仿宋"
        ]
        return random.sample(common_fonts, random.randint(10, 15))
    
    def generate(self) -> Dict[str, Any]:
        """生成完整的浏览器指纹"""
        return {
            "screen_resolution": self.generate_screen_resolution(),
            "timezone": self.generate_timezone(),
            "language": self.generate_language(),
            "platform": self.generate_platform(),
            "webgl_vendor": self.generate_webgl_vendor(),
            "webgl_renderer": self.generate_webgl_renderer(),
            "canvas_fingerprint": self.generate_canvas_fingerprint(),
            "audio_fingerprint": self.generate_audio_fingerprint(),
            "fonts": self.generate_fonts(),
            "hardware_concurrency": random.choice([4, 8, 12, 16]),
            "device_memory": random.choice([4, 8, 16, 32]),
            "touch_points": random.choice([1, 2, 5, 10]),
        }


class RequestFingerprint:
    """请求指纹生成器"""
    
    @staticmethod
    def generate_request_id() -> str:
        """生成请求 ID"""
        timestamp = str(int(time.time() * 1000))
        random_str = str(random.random())
        return hashlib.md5(f"{timestamp}{random_str}".encode()).hexdigest()
    
    @staticmethod
    def generate_session_id() -> str:
        """生成会话 ID"""
        timestamp = str(int(time.time() * 1000))
        random_str = str(random.random())
        return hashlib.sha256(f"{timestamp}{random_str}".encode()).hexdigest()
    
    @staticmethod
    def generate_client_id() -> str:
        """生成客户端 ID"""
        # 使用设备信息生成相对固定的客户端 ID
        device_info = f"{BrowserFingerprint.generate_screen_resolution()}{BrowserFingerprint.generate_platform()}"
        return hashlib.md5(device_info.encode()).hexdigest()
    
    @staticmethod
    def generate_etag() -> str:
        """生成 ETag"""
        return f'"{hashlib.md5(str(time.time()).encode()).hexdigest()[:16]}"'
    
    @staticmethod
    def generate_if_none_match() -> str:
        """生成 If-None-Match"""
        return f'"{hashlib.md5(str(time.time()).encode()).hexdigest()[:16]}"'


class CookiePool:
    """Cookie 池管理"""
    
    def __init__(self, max_cookies_per_domain: int = 10):
        self.cookies: Dict[str, List[Dict[str, Any]]] = {}
        self.max_cookies = max_cookies_per_domain
        self.usage_count: Dict[str, int] = {}
    
    def add_cookie(self, domain: str, cookie: Dict[str, Any]):
        """添加 Cookie"""
        if domain not in self.cookies:
            self.cookies[domain] = []
            self.usage_count[domain] = 0
        
        # 检查是否已存在
        for existing in self.cookies[domain]:
            if existing.get("name") == cookie.get("name"):
                existing.update(cookie)
                return
        
        # 添加新 Cookie
        if len(self.cookies[domain]) >= self.max_cookies:
            # 移除最旧的
            self.cookies[domain].pop(0)
        
        self.cookies[domain].append(cookie)
        logger.debug(f"添加 Cookie 到 {domain}: {cookie.get('name')}")
    
    def get_cookies(self, domain: str) -> Dict[str, str]:
        """获取指定域名的 Cookies"""
        if domain not in self.cookies or not self.cookies[domain]:
            return {}
        
        # 随机选择一个 Cookie 组合
        cookie_dict = {}
        for cookie in self.cookies[domain]:
            cookie_dict[cookie.get("name", "")] = cookie.get("value", "")
        
        self.usage_count[domain] = self.usage_count.get(domain, 0) + 1
        return cookie_dict
    
    def clear(self, domain: Optional[str] = None):
        """清除 Cookies"""
        if domain:
            self.cookies[domain] = []
            self.usage_count[domain] = 0
        else:
            self.cookies.clear()
            self.usage_count.clear()


class AntiCrawlingManager:
    """防爬管理器"""
    
    def __init__(self):
        self.ua_pool = UserAgentPool()
        self.browser_fingerprint = BrowserFingerprint()
        self.request_fingerprint = RequestFingerprint()
        self.cookie_pool = CookiePool()
        self.last_request_time: Dict[str, float] = {}
        self.request_count: Dict[str, int] = {}
    
    def generate_headers(self, url: str, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """生成请求头"""
        # 获取随机 UA
        ua = self.ua_pool.get_random()
        
        # 生成浏览器指纹
        fingerprint = self.browser_fingerprint.generate()
        
        # 基础请求头
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': fingerprint['language'],
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # 添加额外的浏览器指纹信息
        if fingerprint['platform']:
            headers['Sec-Ch-Ua-Platform'] = f'"{fingerprint["platform"]}"'
        
        # 添加自定义请求头
        if custom_headers:
            headers.update(custom_headers)
        
        # 记录使用的 UA
        self.ua_pool.record_success(ua)
        
        logger.debug(f"生成请求头：{url}")
        return headers
    
    def get_cookies(self, url: str) -> Dict[str, str]:
        """获取 Cookies"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        
        return self.cookie_pool.get_cookies(domain)
    
    def add_cookie(self, url: str, name: str, value: str, **kwargs):
        """添加 Cookie"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        
        cookie = {
            "name": name,
            "value": value,
            "domain": domain,
            **kwargs
        }
        self.cookie_pool.add_cookie(domain, cookie)
    
    def should_delay(self, url: str, min_delay: float = 1.0) -> float:
        """计算是否需要延迟"""
        current_time = time.time()
        last_time = self.last_request_time.get(url, 0)
        
        elapsed = current_time - last_time
        if elapsed < min_delay:
            delay = min_delay - elapsed
            logger.debug(f"请求延迟：{delay:.2f}秒")
            return delay
        
        return 0
    
    def record_request(self, url: str):
        """记录请求"""
        self.last_request_time[url] = time.time()
        self.request_count[url] = self.request_count.get(url, 0) + 1
    
    def generate_request_id(self) -> str:
        """生成请求 ID"""
        return self.request_fingerprint.generate_request_id()
    
    def generate_session_id(self) -> str:
        """生成会话 ID"""
        return self.request_fingerprint.generate_session_id()


# 全局防爬管理器实例
_anti_crawling_manager: Optional[AntiCrawlingManager] = None


def get_anti_crawling_manager() -> AntiCrawlingManager:
    """获取防爬管理器实例"""
    global _anti_crawling_manager
    
    if _anti_crawling_manager is None:
        _anti_crawling_manager = AntiCrawlingManager()
    
    return _anti_crawling_manager
