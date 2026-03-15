"""基于 AKShare 的金融数据采集器

此模块基于 AKShare 库实现金融数据采集，支持：
- A 股股票基础数据
- 股票实时行情
- 股票历史行情
- 财务指标数据
- 宏观经济数据

参考：
- AKShare: https://github.com/akfamily/akshare
- ScrapeGraphAI: https://github.com/ScrapeGraphAI/Scrapegraph-ai
"""
import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from app.core.logger import get_logger
from app.services.crawler.base_crawler import AsyncBaseCrawler, RequestConfig, CrawlerStatus

logger = get_logger("akshare_crawler")


@dataclass
class StockData:
    """股票数据结构"""
    ts_code: str  # 股票代码
    name: str  # 股票名称
    trade_date: Optional[date] = None  # 交易日期
    open: Optional[Decimal] = None  # 开盘价
    high: Optional[Decimal] = None  # 最高价
    low: Optional[Decimal] = None  # 最低价
    close: Optional[Decimal] = None  # 收盘价
    volume: Optional[Decimal] = None  # 成交量
    amount: Optional[Decimal] = None  # 成交额
    amplitude: Optional[Decimal] = None  # 振幅
    pct_change: Optional[Decimal] = None  # 涨跌幅
    change_amount: Optional[Decimal] = None  # 涨跌额
    turnover_rate: Optional[Decimal] = None  # 换手率
    pe_ratio: Optional[Decimal] = None  # 市盈率
    pb_ratio: Optional[Decimal] = None  # 市净率
    total_mv: Optional[Decimal] = None  # 总市值
    circ_mv: Optional[Decimal] = None  # 流通市值
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ts_code": self.ts_code,
            "name": self.name,
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "open": float(self.open) if self.open else None,
            "high": float(self.high) if self.high else None,
            "low": float(self.low) if self.low else None,
            "close": float(self.close) if self.close else None,
            "volume": float(self.volume) if self.volume else None,
            "amount": float(self.amount) if self.amount else None,
            "amplitude": float(self.amplitude) if self.amplitude else None,
            "pct_change": float(self.pct_change) if self.pct_change else None,
            "change_amount": float(self.change_amount) if self.change_amount else None,
            "turnover_rate": float(self.turnover_rate) if self.turnover_rate else None,
            "pe_ratio": float(self.pe_ratio) if self.pe_ratio else None,
            "pb_ratio": float(self.pb_ratio) if self.pb_ratio else None,
            "total_mv": float(self.total_mv) if self.total_mv else None,
            "circ_mv": float(self.circ_mv) if self.circ_mv else None,
            **self.extra_data
        }


@dataclass
class FinancialIndicator:
    """财务指标数据结构"""
    ts_code: str
    ann_date: Optional[date] = None  # 公告日期
    end_date: Optional[date] = None  # 报告期
    eps: Optional[Decimal] = None  # 每股收益
    eps_deduct: Optional[Decimal] = None  # 扣非每股收益
    revenue: Optional[Decimal] = None  # 营业总收入
    revenue_yoy: Optional[Decimal] = None  # 营收同比增长率
    net_profit: Optional[Decimal] = None  # 净利润
    net_profit_yoy: Optional[Decimal] = None  # 净利润同比增长率
    roe: Optional[Decimal] = None  # 净资产收益率
    roa: Optional[Decimal] = None  # 总资产收益率
    gross_margin: Optional[Decimal] = None  # 销售毛利率
    net_margin: Optional[Decimal] = None  # 销售净利率
    current_ratio: Optional[Decimal] = None  # 流动比率
    quick_ratio: Optional[Decimal] = None  # 速动比率
    asset_liability_ratio: Optional[Decimal] = None  # 资产负债率
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ts_code": self.ts_code,
            "ann_date": self.ann_date.isoformat() if self.ann_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "eps": float(self.eps) if self.eps else None,
            "eps_deduct": float(self.eps_deduct) if self.eps_deduct else None,
            "revenue": float(self.revenue) if self.revenue else None,
            "revenue_yoy": float(self.revenue_yoy) if self.revenue_yoy else None,
            "net_profit": float(self.net_profit) if self.net_profit else None,
            "net_profit_yoy": float(self.net_profit_yoy) if self.net_profit_yoy else None,
            "roe": float(self.roe) if self.roe else None,
            "roa": float(self.roa) if self.roa else None,
            "gross_margin": float(self.gross_margin) if self.gross_margin else None,
            "net_margin": float(self.net_margin) if self.net_margin else None,
            "current_ratio": float(self.current_ratio) if self.current_ratio else None,
            "quick_ratio": float(self.quick_ratio) if self.quick_ratio else None,
            "asset_liability_ratio": float(self.asset_liability_ratio) if self.asset_liability_ratio else None,
            **self.extra_data
        }


class AKShareCrawler(AsyncBaseCrawler):
    """基于 AKShare 的金融数据采集器"""
    
    def __init__(
        self,
        db_session=None,
        data_callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        初始化 AKShare 爬虫
        
        Args:
            db_session: 数据库会话
            data_callback: 数据回调函数，用于处理采集到的数据
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self.db_session = db_session
        self.data_callback = data_callback
        self._akshare = None
    
    def _import_akshare(self):
        """延迟导入 AKShare"""
        if self._akshare is None:
            try:
                import akshare as ak
                self._akshare = ak
                logger.info("AKShare 导入成功")
            except ImportError as e:
                logger.error(f"AKShare 导入失败：{e}")
                raise ImportError("请安装 AKShare: pip install akshare")
        return self._akshare
    
    async def fetch_stock_basic_info(
        self,
        market: str = "A 股",
        exchange: Optional[str] = None
    ) -> List[StockData]:
        """
        获取股票基础信息
        
        Args:
            market: 市场类型（A 股/港股/美股）
            exchange: 交易所（SH/SZ）
            
        Returns:
            股票基础信息列表
        """
        try:
            ak = self._import_akshare()
            
            # 使用异步执行
            def _fetch():
                if market == "A 股":
                    # 获取 A 股股票列表
                    df = ak.stock_info_a_code_name()
                elif market == "港股":
                    df = ak.stock_hk_spot()
                elif market == "美股":
                    df = ak.stock_us_spot()
                else:
                    logger.warning(f"不支持的市场类型：{market}")
                    return []
                
                return df
            
            # 在线程池中执行
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, _fetch)
            
            # 转换为 StockData 对象
            stocks = []
            for _, row in df.iterrows():
                stock = StockData(
                    ts_code=str(row.get("code", row.get("symbol", ""))),
                    name=str(row.get("name", "")),
                    extra_data=row.to_dict()
                )
                stocks.append(stock)
            
            logger.info(f"获取股票基础信息：成功{len(stocks)}只股票")
            return stocks
            
        except Exception as e:
            logger.error(f"获取股票基础信息失败：{e}")
            return []
    
    async def fetch_stock_realtime_quote(
        self,
        ts_code: str
    ) -> Optional[StockData]:
        """
        获取股票实时行情
        
        Args:
            ts_code: 股票代码
            
        Returns:
            股票实时行情数据
        """
        try:
            ak = self._import_akshare()
            
            def _fetch():
                # 获取实时行情
                df = ak.stock_zh_a_spot_em()
                stock_row = df[df["代码"] == ts_code]
                
                if stock_row.empty:
                    return None
                
                row = stock_row.iloc[0]
                return StockData(
                    ts_code=ts_code,
                    name=str(row.get("名称", "")),
                    open=Decimal(str(row.get("开盘", 0))) if row.get("开盘") else None,
                    high=Decimal(str(row.get("最高", 0))) if row.get("最高") else None,
                    low=Decimal(str(row.get("最低", 0))) if row.get("最低") else None,
                    close=Decimal(str(row.get("最新价", 0))) if row.get("最新价") else None,
                    volume=Decimal(str(row.get("成交量", 0))) if row.get("成交量") else None,
                    amount=Decimal(str(row.get("成交额", 0))) if row.get("成交额") else None,
                    amplitude=Decimal(str(row.get("振幅", 0))) if row.get("振幅") else None,
                    pct_change=Decimal(str(row.get("涨跌幅", 0))) if row.get("涨跌幅") else None,
                    change_amount=Decimal(str(row.get("涨跌额", 0))) if row.get("涨跌额") else None,
                    turnover_rate=Decimal(str(row.get("换手率", 0))) if row.get("换手率") else None,
                    pe_ratio=Decimal(str(row.get("市盈率 - 动态", 0))) if row.get("市盈率 - 动态") else None,
                    pb_ratio=Decimal(str(row.get("市净率", 0))) if row.get("市净率") else None,
                    total_mv=Decimal(str(row.get("总市值", 0))) if row.get("总市值") else None,
                    circ_mv=Decimal(str(row.get("流通市值", 0))) if row.get("流通市值") else None,
                    trade_date=date.today()
                )
            
            loop = asyncio.get_event_loop()
            stock_data = await loop.run_in_executor(None, _fetch)
            
            if stock_data:
                logger.info(f"获取实时行情：{ts_code}")
                # 调用数据回调
                if self.data_callback:
                    await self.data_callback(stock_data.to_dict())
            
            return stock_data
            
        except Exception as e:
            logger.error(f"获取实时行情失败：{ts_code}, 错误：{e}")
            return None
    
    async def fetch_stock_history(
        self,
        ts_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        period: str = "daily"
    ) -> List[StockData]:
        """
        获取股票历史行情
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期（daily/weekly/monthly）
            
        Returns:
            历史行情数据列表
        """
        try:
            ak = self._import_akshare()
            
            # 默认获取最近 1 年的数据
            if not start_date:
                start_date = date.today() - timedelta(days=365)
            if not end_date:
                end_date = date.today()
            
            def _fetch():
                # 获取历史行情
                df = ak.stock_zh_a_hist(
                    symbol=ts_code,
                    period=period,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="qfq"  # 前复权
                )
                
                return df
            
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, _fetch)
            
            # 转换为 StockData 对象
            stocks = []
            for _, row in df.iterrows():
                trade_date = datetime.strptime(str(row.get("日期", "")), "%Y-%m-%d").date()
                
                stock = StockData(
                    ts_code=ts_code,
                    trade_date=trade_date,
                    open=Decimal(str(row.get("开盘", 0))) if row.get("开盘") else None,
                    high=Decimal(str(row.get("最高", 0))) if row.get("最高") else None,
                    low=Decimal(str(row.get("最低", 0))) if row.get("最低") else None,
                    close=Decimal(str(row.get("收盘", 0))) if row.get("收盘") else None,
                    volume=Decimal(str(row.get("成交量", 0))) if row.get("成交量") else None,
                    amount=Decimal(str(row.get("成交额", 0))) if row.get("成交额") else None,
                    amplitude=Decimal(str(row.get("振幅", 0))) if row.get("振幅") else None,
                    pct_change=Decimal(str(row.get("涨跌幅", 0))) if row.get("涨跌幅") else None,
                    change_amount=Decimal(str(row.get("涨跌额", 0))) if row.get("涨跌额") else None,
                    turnover_rate=Decimal(str(row.get("换手率", 0))) if row.get("换手率") else None
                )
                stocks.append(stock)
            
            logger.info(f"获取历史行情：{ts_code}, {len(stocks)}条记录")
            
            # 批量调用数据回调
            if self.data_callback:
                for stock in stocks:
                    await self.data_callback(stock.to_dict())
            
            return stocks
            
        except Exception as e:
            logger.error(f"获取历史行情失败：{ts_code}, 错误：{e}")
            return []
    
    async def fetch_financial_indicators(
        self,
        ts_code: str,
        start_year: int = 2020,
        end_year: int = 2024
    ) -> List[FinancialIndicator]:
        """
        获取财务指标数据
        
        Args:
            ts_code: 股票代码
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            财务指标数据列表
        """
        try:
            ak = self._import_akshare()
            
            def _fetch():
                # 获取财务指标
                df = ak.stock_financial_analysis_indicator(
                    symbol=ts_code,
                    start_year=str(start_year),
                    end_year=str(end_year)
                )
                
                return df
            
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, _fetch)
            
            # 转换为 FinancialIndicator 对象
            indicators = []
            for _, row in df.iterrows():
                try:
                    ann_date = datetime.strptime(str(row.get("公告日期", "")), "%Y-%m-%d").date()
                except:
                    ann_date = None
                
                try:
                    end_date = datetime.strptime(str(row.get("报告期", "")), "%Y-%m-%d").date()
                except:
                    end_date = None
                
                indicator = FinancialIndicator(
                    ts_code=ts_code,
                    ann_date=ann_date,
                    end_date=end_date,
                    eps=Decimal(str(row.get("每股收益 - 基本", 0))) if row.get("每股收益 - 基本") else None,
                    eps_deduct=Decimal(str(row.get("每股收益 - 扣非", 0))) if row.get("每股收益 - 扣非") else None,
                    revenue=Decimal(str(row.get("营业总收入", 0))) if row.get("营业总收入") else None,
                    revenue_yoy=Decimal(str(row.get("营业总收入同比增长", 0))) if row.get("营业总收入同比增长") else None,
                    net_profit=Decimal(str(row.get("归属净利润", 0))) if row.get("归属净利润") else None,
                    net_profit_yoy=Decimal(str(row.get("归属净利润同比增长", 0))) if row.get("归属净利润同比增长") else None,
                    roe=Decimal(str(row.get("净资产收益率 - 加权", 0))) if row.get("净资产收益率 - 加权") else None,
                    roa=Decimal(str(row.get("总资产收益率 - 平均", 0))) if row.get("总资产收益率 - 平均") else None,
                    gross_margin=Decimal(str(row.get("销售毛利率", 0))) if row.get("销售毛利率") else None,
                    net_margin=Decimal(str(row.get("销售净利率", 0))) if row.get("销售净利率") else None,
                    current_ratio=Decimal(str(row.get("流动比率", 0))) if row.get("流动比率") else None,
                    quick_ratio=Decimal(str(row.get("速动比率", 0))) if row.get("速动比率") else None,
                    asset_liability_ratio=Decimal(str(row.get("资产负债率", 0))) if row.get("资产负债率") else None,
                    extra_data=row.to_dict()
                )
                indicators.append(indicator)
            
            logger.info(f"获取财务指标：{ts_code}, {len(indicators)}条记录")
            
            # 调用数据回调
            if self.data_callback:
                for indicator in indicators:
                    await self.data_callback(indicator.to_dict())
            
            return indicators
            
        except Exception as e:
            logger.error(f"获取财务指标失败：{ts_code}, 错误：{e}")
            return []
    
    async def crawl(
        self,
        config: Dict[str, Any]
    ) -> CrawlerStatus:
        """
        执行爬虫任务
        
        Args:
            config: 爬虫配置
            
        Returns:
            爬虫状态
        """
        self.status = CrawlerStatus(
            status="running",
            progress=0,
            items_collected=0,
            current_step="初始化"
        )
        
        try:
            crawler_type = config.get("type", "stock_basic")
            symbols = config.get("symbols", [])
            
            # 解析股票代码列表
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]
            
            total = len(symbols)
            collected = 0
            
            if crawler_type == "stock_basic":
                # 采集股票基础信息
                stocks = await self.fetch_stock_basic_info(
                    market=config.get("market", "A 股")
                )
                collected = len(stocks)
                
            elif crawler_type == "realtime_quote":
                # 采集实时行情
                for i, symbol in enumerate(symbols):
                    self.status.current_step = f"采集 {symbol} 实时行情"
                    stock_data = await self.fetch_stock_realtime_quote(symbol)
                    if stock_data:
                        collected += 1
                    self.status.progress = int((i + 1) / total * 100)
                    
            elif crawler_type == "history":
                # 采集历史行情
                start_date = config.get("start_date")
                end_date = config.get("end_date")
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                
                for i, symbol in enumerate(symbols):
                    self.status.current_step = f"采集 {symbol} 历史行情"
                    stocks = await self.fetch_stock_history(
                        ts_code=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        period=config.get("period", "daily")
                    )
                    collected += len(stocks)
                    self.status.progress = int((i + 1) / total * 100)
                    
            elif crawler_type == "financial":
                # 采集财务指标
                start_year = config.get("start_year", 2020)
                end_year = config.get("end_year", 2024)
                
                for i, symbol in enumerate(symbols):
                    self.status.current_step = f"采集 {symbol} 财务指标"
                    indicators = await self.fetch_financial_indicators(
                        ts_code=symbol,
                        start_year=start_year,
                        end_year=end_year
                    )
                    collected += len(indicators)
                    self.status.progress = int((i + 1) / total * 100)
            
            self.status.status = "completed"
            self.status.progress = 100
            self.status.items_collected = collected
            self.status.current_step = "采集完成"
            
            logger.info(f"AKShare 爬虫完成：采集{collected}条数据")
            
        except Exception as e:
            self.status.status = "failed"
            self.status.error_message = str(e)
            logger.error(f"AKShare 爬虫失败：{e}")
        
        return self.status
