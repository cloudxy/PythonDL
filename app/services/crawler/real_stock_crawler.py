"""真实股票数据采集实现

此模块提供真实的股票数据采集功能，支持多个数据源。
"""
import requests
import time
import random
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from app.core.logger import get_logger

logger = get_logger("stock_crawler_real")


class RealStockCrawler:
    """真实股票爬虫"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 数据源配置
        self.sources = {
            'sina': 'https://hq.sinajs.cn',
            'tencent': 'https://qt.gtimg.cn',
            'eastmoney': 'https://push2.eastmoney.com'
        }
    
    def get_stock_basics_from_sina(self) -> List[Dict]:
        """从新浪财经获取股票基础信息"""
        try:
            # 获取沪深 A 股列表
            url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
            params = {
                'page': 1,
                'num': 80,
                'sort': 'symbol',
                'asc': 1,
                'node': 'hs_a',
                'symbol': '',
                '_s_r_a': 'page'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_sina_basics(data)
            
            return []
        except Exception as e:
            logger.error(f"从新浪财经获取股票基础信息失败：{e}")
            return []
    
    def get_stock_quotes_from_sina(self, ts_codes: List[str]) -> List[Dict]:
        """从新浪财经获取股票行情数据"""
        try:
            quotes = []
            
            # 批量查询，每次最多 100 个
            batch_size = 100
            for i in range(0, len(ts_codes), batch_size):
                batch_codes = ts_codes[i:i+batch_size]
                
                # 构建代码列表
                code_list = [self._convert_code_for_sina(code) for code in batch_codes]
                codes_str = ','.join(code_list)
                
                url = f"{self.sources['sina']}/list={codes_str}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    parsed = self._parse_sina_quotes(response.text, batch_codes)
                    quotes.extend(parsed)
                
                # 避免请求过快
                time.sleep(0.5)
            
            return quotes
        except Exception as e:
            logger.error(f"从新浪财经获取股票行情失败：{e}")
            return []
    
    def get_historical_data(self, ts_code: str, start_date: str, end_date: str) -> List[Dict]:
        """获取历史行情数据"""
        try:
            # 使用新浪财经 API
            code = self._convert_code_for_sina(ts_code)
            url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            params = {
                'symbol': code,
                'scale': 240,  # 日线
                'ma': 'no',
                'datalen': 1000
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_historical_data(data, ts_code, start_date, end_date)
            
            return []
        except Exception as e:
            logger.error(f"获取历史行情数据失败：{e}")
            return []
    
    def _convert_code_for_sina(self, ts_code: str) -> str:
        """转换代码格式为新浪财经格式"""
        # 000001.SZ -> sz000001
        # 600000.SH -> sh600000
        if ts_code.endswith('.SZ'):
            return 'sz' + ts_code.replace('.SZ', '')
        elif ts_code.endswith('.SH'):
            return 'sh' + ts_code.replace('.SH', '')
        return ts_code
    
    def _parse_sina_basics(self, data: List[Dict]) -> List[Dict]:
        """解析新浪财经基础数据"""
        stocks = []
        for item in data:
            try:
                stock = {
                    'ts_code': f"{item.get('symbol', '')}.{'SZ' if item.get('code', '').startswith('0') or item.get('code', '').startswith('3') else 'SH'}",
                    'symbol': item.get('symbol', ''),
                    'name': item.get('name', ''),
                    'area': item.get('area', ''),
                    'industry': item.get('industry', ''),
                    'list_date': item.get('listdate', ''),
                }
                stocks.append(stock)
            except Exception as e:
                logger.warning(f"解析股票基础数据失败：{item}, 错误：{e}")
        
        logger.info(f"解析到 {len(stocks)} 只股票基础信息")
        return stocks
    
    def _parse_sina_quotes(self, text: str, codes: List[str]) -> List[Dict]:
        """解析新浪财经行情数据"""
        quotes = []
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines):
            if i >= len(codes):
                break
            
            try:
                # var hq_str_sz000001="平安银行，12.50,12.48,12.55,12.60,12.45,12.50,12.49..."
                parts = line.split('=')
                if len(parts) != 2:
                    continue
                
                content = parts[1].strip('"').split(',')
                if len(content) < 32:
                    continue
                
                quote = {
                    'ts_code': codes[i],
                    'trade_date': datetime.now().strftime('%Y%m%d'),
                    'open': float(content[1]) if content[1] else 0,
                    'high': float(content[3]) if content[3] else 0,
                    'low': float(content[4]) if content[4] else 0,
                    'close': float(content[2]) if content[2] else 0,
                    'volume': int(float(content[8]) * 100) if content[8] else 0,
                    'amount': float(content[9]) if content[9] else 0,
                }
                quotes.append(quote)
            except Exception as e:
                logger.warning(f"解析行情数据失败：{line}, 错误：{e}")
        
        return quotes
    
    def _parse_historical_data(self, data: List[Dict], ts_code: str, 
                               start_date: str, end_date: str) -> List[Dict]:
        """解析历史行情数据"""
        quotes = []
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        for item in data:
            try:
                date_str = item.get('day', '')
                if not date_str:
                    continue
                
                trade_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if start <= trade_date <= end:
                    quote = {
                        'ts_code': ts_code,
                        'trade_date': trade_date.strftime('%Y%m%d'),
                        'open': float(item.get('open', 0)),
                        'high': float(item.get('high', 0)),
                        'low': float(item.get('low', 0)),
                        'close': float(item.get('close', 0)),
                        'volume': int(float(item.get('volume', 0))),
                        'amount': float(item.get('turnover', 0)),
                    }
                    quotes.append(quote)
            except Exception as e:
                logger.warning(f"解析历史数据失败：{item}, 错误：{e}")
        
        return sorted(quotes, key=lambda x: x['trade_date'])


# 全局爬虫实例
real_stock_crawler = RealStockCrawler()
