"""AI 驱动的智能爬虫基类

此模块提供基于 LLM 的智能爬虫基类，支持自然语言配置和自动数据提取。
参考 ScrapeGraphAI 架构设计。
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
from aiohttp import ClientTimeout, TCPConnector

from app.core.logger import get_logger
from app.core.redis_client import get_redis_client
from app.services.crawler.anti_crawling import get_anti_crawling_manager
from app.services.crawler.request_dedup import get_deduplicator
from app.services.crawler.llm_service import LLMServiceFactory, BaseLLMService

logger = get_logger("ai_crawler")

T = TypeVar('T')


class LLMProvider(str, Enum):
    """LLM 提供商"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"
    AZURE = "azure"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.2"
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_tokens: int = 8192
    format: str = "json"  # json, text
    temperature: float = 0.0
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "api_key": self.api_key,
            "api_url": self.api_url,
            "model_tokens": self.model_tokens,
            "format": self.format,
            "temperature": self.temperature,
            "verbose": self.verbose,
        }


@dataclass
class CrawlerGraphConfig:
    """爬虫图配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    verbose: bool = True
    headless: bool = False
    max_results: int = 10
    cache_enabled: bool = True
    dedup_enabled: bool = True
    proxy_enabled: bool = False
    rate_limit_delay: float = 0.1
    max_concurrent: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "llm": self.llm.to_dict(),
            "verbose": self.verbose,
            "headless": self.headless,
            "max_results": self.max_results,
            "cache_enabled": self.cache_enabled,
            "dedup_enabled": self.dedup_enabled,
            "proxy_enabled": self.proxy_enabled,
            "rate_limit_delay": self.rate_limit_delay,
            "max_concurrent": self.max_concurrent,
        }


@dataclass
class CrawlerExecutionState:
    """爬虫执行状态"""
    id: str = ""
    name: str = ""
    status: str = "pending"  # pending, running, completed, failed
    progress: int = 0
    current_step: str = ""
    total_steps: int = 0
    items_extracted: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "items_extracted": self.items_extracted,
            "errors": self.errors,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": self.result,
        }


class AIBaseCrawler(ABC, Generic[T]):
    """AI 驱动的基础爬虫类
    
    基于 ScrapeGraphAI 理念设计，支持：
    - 自然语言 prompt 配置
    - LLM 驱动的数据提取
    - 多种爬虫图模式
    - 实时状态监控
    """
    
    def __init__(
        self,
        prompt: str,
        config: Optional[CrawlerGraphConfig] = None,
        crawler_type: str = "ai_base"
    ):
        self.prompt = prompt
        self.config = config or CrawlerGraphConfig()
        self.crawler_type = crawler_type
        
        # HTTP 会话
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector = TCPConnector(
            limit=self.config.max_concurrent,
            limit_per_host=self.config.max_concurrent // 2,
        )
        
        # 执行状态
        self.state = CrawlerExecutionState()
        
        # 防爬组件
        self.anti_crawling = None
        self.deduplicator = None
        
        # 信号量
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # LLM 服务
        self.llm_service: Optional[BaseLLMService] = None
    
    async def initialize(self):
        """初始化爬虫"""
        try:
            # 创建 HTTP 会话
            timeout = ClientTimeout(total=120)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers=self._get_default_headers()
            )
            
            # 初始化防爬组件
            if self.config.dedup_enabled:
                self.deduplicator = get_deduplicator()
                await self.deduplicator.initialize()
            
            if self.config.proxy_enabled:
                self.anti_crawling = get_anti_crawling_manager()
            
            # 初始化 LLM 服务
            self.llm_service = LLMServiceFactory.create_service(
                provider=self.config.llm.provider.value,
                api_key=self.config.llm.api_key,
                api_url=self.config.llm.api_url,
                model=self.config.llm.model
            )
            await self.llm_service.initialize()
            
            logger.info(f"{self.crawler_type} AI 爬虫初始化完成，LLM: {self.config.llm.provider.value}/{self.config.llm.model}")
            
        except Exception as e:
            logger.error(f"{self.crawler_type} AI 爬虫初始化失败：{e}")
            raise
    
    async def close(self):
        """关闭爬虫"""
        if self.session:
            await self.session.close()
        if self.deduplicator:
            await self.deduplicator.close()
        if self.llm_service:
            await self.llm_service.close()
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _get_default_headers(self) -> Dict[str, str]:
        """获取默认请求头"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
    
    def _update_state(
        self,
        status: str,
        progress: int,
        current_step: str,
        items_extracted: int = 0,
        error: Optional[str] = None
    ):
        """更新执行状态"""
        self.state.status = status
        self.state.progress = progress
        self.state.current_step = current_step
        self.state.items_extracted = items_extracted
        
        if error:
            self.state.errors.append(error)
        
        logger.info(f"[{self.state.name}] {current_step} - 进度：{progress}%")
    
    async def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Optional[aiohttp.ClientResponse]:
        """发送 HTTP 请求"""
        async with self.semaphore:
            try:
                request_headers = self._get_default_headers()
                if headers:
                    request_headers.update(headers)
                
                async with self.session.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=request_headers
                ) as response:
                    response.raise_for_status()
                    return response
            except Exception as e:
                logger.error(f"请求失败 {url}: {e}")
                return None
    
    async def _call_llm(self, prompt: str, context: str) -> Dict[str, Any]:
        """调用 LLM 进行数据提取"""
        if not self.llm_service:
            raise RuntimeError("LLM 服务未初始化")
        
        try:
            # 使用 JSON 格式输出
            if self.config.llm.format == "json":
                system_prompt = "你是一个数据提取助手。请从提供的内容中提取信息，并严格按照 JSON 格式返回结果。不要包含任何解释，只返回 JSON。"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                result = await self.llm_service.chat(
                    messages,
                    temperature=self.config.llm.temperature,
                )
                
                # 解析结果
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if response_text:
                    # 清理 Markdown 格式
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    return json.loads(response_text.strip())
                return {}
            else:
                response = await self.llm_service.generate(prompt)
                return {"text": response}
                
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return {}
    
    async def _extract_data(self, html: str, prompt: str) -> Dict[str, Any]:
        """使用 LLM 从 HTML 中提取数据"""
        # 构建 LLM prompt
        llm_prompt = f"""
        请从以下 HTML 内容中提取我需要的信息。
        
        提取要求：{prompt}
        
        HTML 内容：
        {html[:5000]}  # 限制 HTML 长度
        
        请以 JSON 格式返回提取的结果。
        """
        
        try:
            result = await self._call_llm(llm_prompt, html)
            return result
        except Exception as e:
            logger.error(f"LLM 数据提取失败：{e}")
            return {}
    
    @abstractmethod
    async def crawl(self, **kwargs) -> List[T]:
        """爬虫入口方法"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """获取执行状态"""
        return self.state.to_dict()


class SmartScraperCrawler(AIBaseCrawler[T]):
    """智能单页爬虫
    
    类似 ScrapeGraphAI 的 SmartScraperGraph，只需提供 prompt 和 URL
    """
    
    def __init__(
        self,
        prompt: str,
        source: str,
        config: Optional[CrawlerGraphConfig] = None
    ):
        super().__init__(prompt=prompt, config=config, crawler_type="smart_scraper")
        self.source = source
        self.state.id = f"smart_scraper_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.state.name = f"SmartScraper: {source}"
        self.state.total_steps = 3
    
    async def crawl(self, **kwargs) -> List[T]:
        """执行智能爬取"""
        try:
            self._update_state("running", 0, "正在连接目标网站...")
            
            # 步骤 1: 获取网页内容
            self._update_state("running", 10, "正在下载网页内容...")
            response = await self._make_request(self.source)
            
            if not response:
                self._update_state("failed", 0, "无法访问目标网站", error="请求失败")
                return []
            
            html = await response.text()
            
            # 步骤 2: 使用 LLM 提取数据
            self._update_state("running", 40, "正在使用 AI 提取数据...")
            extracted_data = await self._extract_data(html, self.prompt)
            
            # 步骤 3: 处理和验证数据
            self._update_state("running", 80, "正在验证和格式化数据...")
            result = await self._process_extracted_data(extracted_data)
            
            # 完成
            self._update_state("completed", 100, "爬取完成", items_extracted=len(result))
            self.state.result = {"data": result}
            
            return result
            
        except Exception as e:
            self._update_state("failed", 0, "爬取失败", error=str(e))
            logger.error(f"SmartScraper 爬取失败：{e}")
            return []
    
    async def _process_extracted_data(self, data: Dict[str, Any]) -> List[T]:
        """处理提取的数据"""
        if not data:
            return []
        
        # 如果是列表，直接返回
        if isinstance(data, list):
            return data
        
        # 如果是字典，包装成列表
        return [data]
    
    async def _call_llm(self, prompt: str, context: str) -> Dict[str, Any]:
        """调用 LLM"""
        # 这里需要根据实际的 LLM 提供商实现
        # 暂时返回模拟数据
        logger.info(f"调用 LLM: {self.config.llm.provider.value}/{self.config.llm.model}")
        
        # TODO: 实现真实的 LLM 调用
        # 对于 Ollama:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         "http://localhost:11434/api/generate",
        #         json={
        #             "model": self.config.llm.model,
        #             "prompt": prompt,
        #             "stream": False,
        #             "format": "json"
        #         }
        #     ) as response:
        #         result = await response.json()
        #         return json.loads(result.get("response", "{}"))
        
        return {}


class SearchGraphCrawler(AIBaseCrawler[T]):
    """搜索引擎爬虫
    
    类似 ScrapeGraphAI 的 SearchGraph，从搜索引擎获取多个结果
    """
    
    def __init__(
        self,
        prompt: str,
        query: str,
        search_engine: str = "google",
        max_results: int = 10,
        config: Optional[CrawlerGraphConfig] = None
    ):
        super().__init__(prompt=prompt, config=config, crawler_type="search_graph")
        self.query = query
        self.search_engine = search_engine
        self.max_results = max_results
        self.state.id = f"search_graph_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.state.name = f"SearchGraph: {query}"
        self.state.total_steps = 4
    
    async def crawl(self, **kwargs) -> List[T]:
        """执行搜索爬取"""
        try:
            # 步骤 1: 执行搜索
            self._update_state("running", 0, f"正在 {self.search_engine} 搜索...")
            search_results = await self._perform_search(self.query)
            
            # 步骤 2: 爬取搜索结果
            self._update_state("running", 30, f"正在爬取 {len(search_results)} 个搜索结果...")
            all_data = []
            
            for i, result in enumerate(search_results[:self.max_results]):
                self._update_state(
                    "running",
                    30 + int((i / len(search_results)) * 50),
                    f"正在处理结果 {i+1}/{len(search_results)}..."
                )
                
                data = await self._scrape_search_result(result)
                if data:
                    all_data.append(data)
            
            # 步骤 3: 汇总结果
            self._update_state("running", 90, "正在汇总结果...")
            combined_result = await self._combine_results(all_data)
            
            # 完成
            self._update_state("completed", 100, "搜索完成", items_extracted=len(combined_result))
            self.state.result = {"data": combined_result}
            
            return combined_result
            
        except Exception as e:
            self._update_state("failed", 0, "搜索失败", error=str(e))
            logger.error(f"SearchGraph 爬取失败：{e}")
            return []
    
    async def _perform_search(self, query: str) -> List[Dict[str, str]]:
        """执行搜索"""
        # TODO: 实现真实的搜索引擎调用
        # 这里需要根据不同的搜索引擎实现
        logger.info(f"执行搜索：{query} @ {self.search_engine}")
        return []
    
    async def _scrape_search_result(self, result: Dict[str, str]) -> Optional[T]:
        """爬取单个搜索结果"""
        url = result.get("url")
        if not url:
            return None
        
        response = await self._make_request(url)
        if not response:
            return None
        
        html = await response.text()
        return await self._extract_data(html, self.prompt)
    
    async def _combine_results(self, results: List[T]) -> List[T]:
        """合并结果"""
        return results
    
    async def _call_llm(self, prompt: str, context: str) -> Dict[str, Any]:
        """调用 LLM"""
        logger.info(f"调用 LLM: {self.config.llm.provider.value}/{self.config.llm.model}")
        return {}
