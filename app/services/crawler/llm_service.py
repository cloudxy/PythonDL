"""LLM 服务集成

此模块提供与各种 LLM 服务的集成，包括 Ollama、OpenAI、Groq 等。
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum

import aiohttp

from app.core.logger import get_logger

logger = get_logger("llm_service")


class LLMProviderType(str, Enum):
    """LLM 提供商类型"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"
    AZURE = "azure"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class BaseLLMService(ABC):
    """LLM 服务基类"""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """初始化"""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """关闭"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """发送聊天请求"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成 JSON 格式结果"""
        response = await self.generate(prompt, **kwargs)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}")
            return {}


class OllamaService(BaseLLMService):
    """Ollama LLM 服务"""
    
    def __init__(self, api_url: str = "http://localhost:11434", model: str = "llama3.2"):
        super().__init__(api_url=api_url)
        self.model = model
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """聊天接口"""
        if not self.session:
            await self.initialize()
        
        url = f"{self.api_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        async with self.session.post(url, json=payload) as response:
            result = await response.json()
            return result
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成接口"""
        if not self.session:
            await self.initialize()
        
        url = f"{self.api_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        async with self.session.post(url, json=payload) as response:
            result = await response.json()
            return result.get("response", "")


class OpenAIService(BaseLLMService):
    """OpenAI LLM 服务"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(api_key=api_key)
        self.model = model
        self.api_url = "https://api.openai.com/v1"
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """聊天接口"""
        if not self.session:
            await self.initialize()
        
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        async with self.session.post(url, json=payload, headers=headers) as response:
            result = await response.json()
            return result
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成接口（通过聊天接口实现）"""
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat(messages, **kwargs)
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


class GroqService(BaseLLMService):
    """Groq LLM 服务"""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        super().__init__(api_key=api_key)
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1"
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """聊天接口"""
        if not self.session:
            await self.initialize()
        
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        async with self.session.post(url, json=payload, headers=headers) as response:
            result = await response.json()
            return result
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成接口"""
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat(messages, **kwargs)
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


class LLMServiceFactory:
    """LLM 服务工厂"""
    
    _services = {}
    
    @classmethod
    def create_service(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None
    ) -> BaseLLMService:
        """创建 LLM 服务实例"""
        provider = provider.lower()
        
        if provider in cls._services:
            return cls._services[provider]
        
        if provider == "ollama":
            service = OllamaService(
                api_url=api_url or "http://localhost:11434",
                model=model or "llama3.2"
            )
        elif provider == "openai":
            if not api_key:
                raise ValueError("OpenAI 需要 API key")
            service = OpenAIService(
                api_key=api_key,
                model=model or "gpt-4o-mini"
            )
        elif provider == "groq":
            if not api_key:
                raise ValueError("Groq 需要 API key")
            service = GroqService(
                api_key=api_key,
                model=model or "llama3-70b-8192"
            )
        else:
            raise ValueError(f"不支持的 LLM 提供商：{provider}")
        
        cls._services[provider] = service
        return service
    
    @classmethod
    async def get_service(
        cls,
        provider: str,
        **kwargs
    ) -> BaseLLMService:
        """获取并初始化 LLM 服务"""
        service = cls.create_service(provider, **kwargs)
        await service.initialize()
        return service
