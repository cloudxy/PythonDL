"""智能金融数据采集器 - 基于 ScrapeGraphAI 架构

此模块结合 ScrapeGraphAI 的智能数据提取能力和 AKShare 的金融数据接口，实现：
- 自然语言配置的金融数据采集
- AI 驱动的网页数据提取
- 多数据源智能融合
- 自动化数据清洗和结构化

参考架构：
- ScrapeGraphAI: https://github.com/ScrapeGraphAI/Scrapegraph-ai
- SmartScraper: LLM 驱动的网页数据提取
- SearchGraph: 多源数据搜索和整合
"""
import asyncio
import json
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from app.core.logger import get_logger
from app.services.crawler.ai_crawler import (
    AIBaseCrawler,
    SmartScraperCrawler,
    CrawlerGraphConfig,
    LLMConfig,
    CrawlerExecutionState
)
from app.services.crawler.akshare_crawler import AKShareCrawler, StockData, FinancialIndicator

logger = get_logger("smart_finance_crawler")


@dataclass
class FinanceDataExtractionResult:
    """金融数据提取结果"""
    source: str  # 数据来源
    data_type: str  # 数据类型
    data: Dict[str, Any]  # 提取的数据
    confidence: float = 1.0  # 置信度
    extraction_time: float = 0.0  # 提取耗时
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "data_type": self.data_type,
            "data": self.data,
            "confidence": self.confidence,
            "extraction_time": self.extraction_time,
            "metadata": self.metadata
        }


class SmartFinanceCrawler(AIBaseCrawler):
    """智能金融数据采集器
    
    结合 AKShare 和 AI 数据提取的混合爬虫
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        graph_config: Optional[CrawlerGraphConfig] = None,
        db_session=None,
        **kwargs
    ):
        """
        初始化智能金融爬虫
        
        Args:
            llm_config: LLM 配置
            graph_config: 爬虫图配置
            db_session: 数据库会话
            **kwargs: 其他参数
        """
        super().__init__(llm_config, graph_config, **kwargs)
        self.db_session = db_session
        
        # 初始化 AKShare 爬虫
        self.akshare_crawler = AKShareCrawler(
            db_session=db_session,
            data_callback=self._on_akshare_data
        )
        
        # 数据缓存
        self.data_cache: Dict[str, Any] = {}
        
        # 数据提取提示词模板
        self.extraction_prompts = {
            "stock_info": """
你是一个金融数据提取专家。请从以下内容中提取股票信息：

需要提取的字段：
- ts_code: 股票代码
- name: 股票名称
- market: 市场（A 股/港股/美股）
- industry: 所属行业
- area: 所属地区
- list_date: 上市日期

请以 JSON 格式返回提取结果。
""",
            
            "stock_quote": """
你是一个金融数据提取专家。请从以下内容中提取股票行情数据：

需要提取的字段：
- ts_code: 股票代码
- trade_date: 交易日期
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量
- amount: 成交额
- amplitude: 振幅
- pct_change: 涨跌幅
- change_amount: 涨跌额
- turnover_rate: 换手率
- pe_ratio: 市盈率
- pb_ratio: 市净率

请以 JSON 格式返回提取结果。
""",
            
            "financial_indicator": """
你是一个金融数据提取专家。请从以下内容中提取财务指标数据：

需要提取的字段：
- ts_code: 股票代码
- ann_date: 公告日期
- end_date: 报告期
- eps: 每股收益
- revenue: 营业总收入
- revenue_yoy: 营收同比增长率
- net_profit: 净利润
- net_profit_yoy: 净利润同比增长率
- roe: 净资产收益率
- roa: 总资产收益率
- gross_margin: 销售毛利率
- net_margin: 销售净利率

请以 JSON 格式返回提取结果。
"""
        }
    
    async def _on_akshare_data(self, data: Dict[str, Any]):
        """AKShare 数据回调"""
        # 可以在这里处理 AKShare 采集到的数据
        logger.debug(f"AKShare 数据：{data.get('ts_code', 'N/A')}")
    
    async def extract_from_web(
        self,
        url: str,
        prompt: str,
        data_type: str = "stock_info"
    ) -> Optional[FinanceDataExtractionResult]:
        """
        从网页提取金融数据
        
        Args:
            url: 网页 URL
            prompt: 提取提示词
            data_type: 数据类型
            
        Returns:
            数据提取结果
        """
        try:
            # 使用 SmartScraper 提取数据
            smart_scraper = SmartScraperCrawler(
                prompt=prompt,
                source=url,
                config=self.graph_config
            )
            
            start_time = datetime.now()
            result = await smart_scraper.run()
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            if result and result.get("data"):
                return FinanceDataExtractionResult(
                    source=url,
                    data_type=data_type,
                    data=result["data"],
                    confidence=result.get("confidence", 1.0),
                    extraction_time=extraction_time,
                    metadata=result.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"网页数据提取失败：{e}")
            return None
    
    async def extract_from_akshare(
        self,
        data_type: str,
        **kwargs
    ) -> List[FinanceDataExtractionResult]:
        """
        从 AKShare 获取金融数据
        
        Args:
            data_type: 数据类型（stock_basic/realtime_quote/history/financial）
            **kwargs: 参数
            
        Returns:
            数据提取结果列表
        """
        results = []
        
        try:
            if data_type == "stock_basic":
                stocks = await self.akshare_crawler.fetch_stock_basic_info(
                    market=kwargs.get("market", "A 股")
                )
                for stock in stocks:
                    results.append(FinanceDataExtractionResult(
                        source="AKShare",
                        data_type="stock_basic",
                        data=stock.to_dict(),
                        confidence=1.0
                    ))
            
            elif data_type == "realtime_quote":
                ts_code = kwargs.get("ts_code")
                if ts_code:
                    stock_data = await self.akshare_crawler.fetch_stock_realtime_quote(ts_code)
                    if stock_data:
                        results.append(FinanceDataExtractionResult(
                            source="AKShare",
                            data_type="realtime_quote",
                            data=stock_data.to_dict(),
                            confidence=1.0
                        ))
            
            elif data_type == "history":
                ts_code = kwargs.get("ts_code")
                start_date = kwargs.get("start_date")
                end_date = kwargs.get("end_date")
                if ts_code:
                    stocks = await self.akshare_crawler.fetch_stock_history(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date
                    )
                    for stock in stocks:
                        results.append(FinanceDataExtractionResult(
                            source="AKShare",
                            data_type="history",
                            data=stock.to_dict(),
                            confidence=1.0
                        ))
            
            elif data_type == "financial":
                ts_code = kwargs.get("ts_code")
                start_year = kwargs.get("start_year", 2020)
                end_year = kwargs.get("end_year", 2024)
                if ts_code:
                    indicators = await self.akshare_crawler.fetch_financial_indicators(
                        ts_code=ts_code,
                        start_year=start_year,
                        end_year=end_year
                    )
                    for indicator in indicators:
                        results.append(FinanceDataExtractionResult(
                            source="AKShare",
                            data_type="financial",
                            data=indicator.to_dict(),
                            confidence=1.0
                        ))
            
        except Exception as e:
            logger.error(f"AKShare 数据提取失败：{e}")
        
        return results
    
    async def extract_multi_source(
        self,
        data_type: str,
        sources: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        多源数据提取和融合
        
        Args:
            data_type: 数据类型
            sources: 数据源配置列表
            **kwargs: 其他参数
            
        Returns:
            融合后的数据
        """
        all_results = []
        
        # 并行采集多个数据源
        tasks = []
        for source in sources:
            if source.get("type") == "akshare":
                task = self.extract_from_akshare(data_type, **kwargs)
                tasks.append(task)
            elif source.get("type") == "web":
                url = source.get("url")
                prompt = source.get("prompt", self.extraction_prompts.get(data_type, ""))
                task = self.extract_from_web(url, prompt, data_type)
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, FinanceDataExtractionResult):
                    all_results.append(result)
        
        # 数据融合
        merged_data = self._merge_data(all_results)
        
        return {
            "data_type": data_type,
            "sources_used": [s.get("type") for s in sources],
            "total_results": len(all_results),
            "merged_data": merged_data,
            "raw_results": [r.to_dict() for r in all_results]
        }
    
    def _merge_data(self, results: List[FinanceDataExtractionResult]) -> Dict[str, Any]:
        """
        数据融合
        
        Args:
            results: 数据提取结果列表
            
        Returns:
            融合后的数据
        """
        if not results:
            return {}
        
        # 按数据来源分组
        by_source = {}
        for result in results:
            source = result.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result.data)
        
        # 简单融合策略：优先使用 AKShare 数据，其他数据源作为补充
        merged = {}
        
        # 首先合并 AKShare 数据
        if "AKShare" in by_source:
            ak_data = by_source["AKShare"]
            if len(ak_data) == 1:
                merged = ak_data[0]
            else:
                # 多条数据，返回列表
                merged["items"] = ak_data
        
        # 合并其他数据源
        for source, data_list in by_source.items():
            if source != "AKShare":
                if "alternative_sources" not in merged:
                    merged["alternative_sources"] = {}
                merged["alternative_sources"][source] = data_list
        
        return merged
    
    async def run(
        self,
        config: Dict[str, Any]
    ) -> CrawlerExecutionState:
        """
        运行智能金融爬虫
        
        Args:
            config: 爬虫配置
            
        Returns:
            爬虫执行状态
        """
        self.state = CrawlerExecutionState(
            id=config.get("id", ""),
            name=config.get("name", "智能金融爬虫"),
            status="running",
            progress=0,
            current_step="初始化"
        )
        
        try:
            data_type = config.get("data_type", "stock_basic")
            sources = config.get("sources", [])
            symbols = config.get("symbols", [])
            
            # 解析股票代码
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]
            
            total_symbols = len(symbols)
            collected = 0
            
            if not sources:
                # 默认使用 AKShare
                sources = [{"type": "akshare"}]
            
            # 遍历股票代码采集
            for i, symbol in enumerate(symbols):
                self.state.current_step = f"采集 {symbol} - {data_type}"
                
                result = await self.extract_multi_source(
                    data_type=data_type,
                    sources=sources,
                    ts_code=symbol,
                    market=config.get("market", "A 股"),
                    start_date=config.get("start_date"),
                    end_date=config.get("end_date"),
                    start_year=config.get("start_year", 2020),
                    end_year=config.get("end_year", 2024)
                )
                
                # 缓存结果
                cache_key = f"{data_type}:{symbol}"
                self.data_cache[cache_key] = result
                
                collected += 1
                self.state.progress = int((i + 1) / total_symbols * 100)
                
                # 调用数据回调
                if self.data_callback and result:
                    await self.data_callback(result)
            
            self.state.status = "completed"
            self.state.progress = 100
            self.state.items_collected = collected
            self.state.current_step = "采集完成"
            
            logger.info(f"智能金融爬虫完成：采集{collected}个股票数据")
            
        except Exception as e:
            self.state.status = "failed"
            self.state.error_message = str(e)
            logger.error(f"智能金融爬虫失败：{e}")
        
        return self.state
    
    def get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存数据"""
        return self.data_cache.get(key)
    
    def clear_cache(self):
        """清空缓存"""
        self.data_cache.clear()
