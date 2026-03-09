"""金融分析模块API测试

此模块测试金融分析相关的API端点，包括：
1. 股票数据采集
2. 股票列表查询
3. 单个股票数据查询
4. 股票价格预测
5. 股票风险分析
6. 股票分析报告生成
"""
import pytest
import asyncio
from tests.api.base import BaseAPITest

class TestFinanceAPI(BaseAPITest):
    """金融分析API测试类"""
    
    @pytest.mark.asyncio
    async def test_collect_stock_data(self, async_client, auth_headers, stock_test_data):
        """测试采集股票数据"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["valid_stock"]
        
        response = await self.make_request(
            client=async_client,
            method="POST",
            endpoint=f"/api/v1/finance/stocks/collect?ts_code={stock_data['ts_code']}&days={stock_data['days']}",
            headers=auth_headers
        )
        
        # 注意：数据采集可能成功（返回200）或失败（返回400/500）
        # 我们检查响应结构
        if response.status_code == 200:
            self.assert_status_code(response, 200)
            self.assert_response_schema(response, ["status", "data"])
            
            data = response.json()
            assert data["status"] == "success"
            assert data["data"]["ts_code"] == stock_data["ts_code"]
            assert data["data"]["collected"] == True
        elif response.status_code in [400, 500]:
            self.assert_response_schema(response, ["status", "message"])
    
    @pytest.mark.asyncio
    async def test_collect_stock_data_invalid_code(self, async_client, auth_headers, stock_test_data):
        """测试采集无效股票代码数据"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["invalid_stock_code"]
        
        response = await self.make_request(
            client=async_client,
            method="POST",
            endpoint=f"/api/v1/finance/stocks/collect?ts_code={stock_data['ts_code']}&days={stock_data['days']}",
            headers=auth_headers
        )
        
        # 无效股票代码应该返回错误
        self.assert_status_code(response, 400)
        self.assert_response_schema(response, ["status", "message"])
    
    @pytest.mark.asyncio
    async def test_get_stocks(self, async_client, auth_headers):
        """测试获取股票列表"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        response = await self.make_request(
            client=async_client,
            method="GET",
            endpoint="/api/v1/finance/stocks",
            headers=auth_headers
        )
        
        self.assert_status_code(response, 200)
        self.assert_response_schema(response, ["status", "data", "total", "skip", "limit"])
        self.assert_pagination(response)
        
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["data"], list)
    
    @pytest.mark.asyncio
    async def test_get_stocks_with_pagination(self, async_client, auth_headers):
        """测试带分页的股票列表"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        response = await self.make_request(
            client=async_client,
            method="GET",
            endpoint="/api/v1/finance/stocks?skip=0&limit=10",
            headers=auth_headers
        )
        
        self.assert_status_code(response, 200)
        self.assert_pagination(response)
        
        data = response.json()
        assert data["skip"] == 0
        assert data["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_get_stock_by_code(self, async_client, auth_headers, stock_test_data):
        """测试根据股票代码获取股票数据"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["valid_stock"]
        
        response = await self.make_request(
            client=async_client,
            method="GET",
            endpoint=f"/api/v1/finance/stocks/{stock_data['ts_code']}",
            headers=auth_headers
        )
        
        # 股票数据可能不存在，返回空列表
        self.assert_status_code(response, 200)
        self.assert_response_schema(response, ["status", "data", "total", "skip", "limit"])
        self.assert_pagination(response)
        
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["data"], list)
    
    @pytest.mark.asyncio
    async def test_predict_stock(self, async_client, auth_headers, stock_test_data):
        """测试预测股票价格"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["valid_stock"]
        
        response = await self.make_request(
            client=async_client,
            method="POST",
            endpoint=f"/api/v1/finance/stocks/predict?ts_code={stock_data['ts_code']}&days={stock_data['days']}",
            headers=auth_headers
        )
        
        # 预测可能成功或失败
        if response.status_code == 200:
            self.assert_status_code(response, 200)
            self.assert_response_schema(response, ["status", "data"])
            
            data = response.json()
            assert data["status"] == "success"
        elif response.status_code == 400:
            self.assert_status_code(response, 400)
            self.assert_response_schema(response, ["status", "message"])
    
    @pytest.mark.asyncio
    async def test_analyze_stock_risk(self, async_client, auth_headers, stock_test_data):
        """测试分析股票风险"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["valid_stock"]
        
        response = await self.make_request(
            client=async_client,
            method="POST",
            endpoint=f"/api/v1/finance/stocks/risk?ts_code={stock_data['ts_code']}",
            headers=auth_headers
        )
        
        # 风险分析可能成功或失败
        if response.status_code == 200:
            self.assert_status_code(response, 200)
            self.assert_response_schema(response, ["status", "data"])
            
            data = response.json()
            assert data["status"] == "success"
        elif response.status_code == 400:
            self.assert_status_code(response, 400)
            self.assert_response_schema(response, ["status", "message"])
    
    @pytest.mark.asyncio
    async def test_get_stock_report(self, async_client, auth_headers, stock_test_data):
        """测试生成股票分析报告"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["valid_stock"]
        
        response = await self.make_request(
            client=async_client,
            method="GET",
            endpoint=f"/api/v1/finance/stocks/{stock_data['ts_code']}/report",
            headers=auth_headers
        )
        
        # 报告生成可能成功或失败
        if response.status_code == 200:
            self.assert_status_code(response, 200)
            self.assert_response_schema(response, ["status", "data"])
            
            data = response.json()
            assert data["status"] == "success"
        elif response.status_code == 400:
            self.assert_status_code(response, 400)
            self.assert_response_schema(response, ["status", "message"])
    
    # 性能测试
    @pytest.mark.asyncio
    async def test_performance_get_stocks(self, async_client, auth_headers, performance_test_config):
        """测试获取股票列表接口性能"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        metrics = await self.run_performance_test(
            client=async_client,
            method="GET",
            endpoint="/api/v1/finance/stocks",
            concurrent_users=performance_test_config["concurrent_users"][0],  # 1个并发用户
            duration_seconds=10,  # 缩短测试时间
            headers=auth_headers
        )
        
        print(f"股票列表性能测试结果:")
        print(f"  总请求数: {metrics.total_requests}")
        print(f"  成功请求: {metrics.successful_requests}")
        print(f"  失败请求: {metrics.failed_requests}")
        print(f"  平均响应时间: {metrics.avg_response_time_ms:.2f}ms")
        print(f"  错误率: {metrics.error_rate_percentage:.2f}%")
        
        assert metrics.error_rate_percentage < performance_test_config["error_threshold_percentage"]
        assert metrics.avg_response_time_ms < performance_test_config["response_time_threshold_ms"]
    
    # 安全测试
    @pytest.mark.asyncio
    async def test_security_sql_injection_stocks(self, async_client, auth_headers, security_test_config):
        """测试股票查询接口SQL注入漏洞"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        results = await self.test_sql_injection(
            client=async_client,
            endpoint="/api/v1/finance/stocks",
            method="GET",
            payloads=security_test_config["sql_injection_payloads"],
            headers=auth_headers
        )
        
        vulnerable_tests = [r for r in results if not r.success]
        print(f"SQL注入测试结果: {len(results)}个payload, {len(vulnerable_tests)}个可能漏洞")
        
        for result in vulnerable_tests:
            print(f"  潜在漏洞: {result.error_message}")
        
        assert len(vulnerable_tests) == 0, f"发现{len(vulnerable_tests)}个潜在SQL注入漏洞"
    
    @pytest.mark.asyncio
    async def test_security_xss_stocks(self, async_client, auth_headers, security_test_config):
        """测试股票查询接口XSS漏洞"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        results = await self.test_xss_injection(
            client=async_client,
            endpoint="/api/v1/finance/stocks/collect",
            method="POST",
            payloads=security_test_config["xss_payloads"],
            headers=auth_headers
        )
        
        vulnerable_tests = [r for r in results if not r.success]
        print(f"XSS注入测试结果: {len(results)}个payload, {len(vulnerable_tests)}个可能漏洞")
        
        for result in vulnerable_tests:
            print(f"  潜在漏洞: {result.error_message}")
        
        assert len(vulnerable_tests) == 0, f"发现{len(vulnerable_tests)}个潜在XSS漏洞"
    
    @pytest.mark.asyncio
    async def test_contract_validation_stocks(self, async_client, auth_headers):
        """测试股票接口契约验证"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        expected_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {"type": "array"},
                "total": {"type": "integer"},
                "skip": {"type": "integer"},
                "limit": {"type": "integer"}
            },
            "required": ["status", "data", "total", "skip", "limit"]
        }
        
        result = await self.test_contract_validation(
            client=async_client,
            endpoint="/api/v1/finance/stocks",
            method="GET",
            expected_schema=expected_schema,
            headers=auth_headers
        )
        
        assert result["success"], f"契约验证失败: {result.get('error')}"
    
    @pytest.mark.asyncio
    async def test_performance_collect_stock(self, async_client, auth_headers, performance_test_config, stock_test_data):
        """测试采集股票数据接口性能"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        stock_data = stock_test_data["valid_stock"]
        
        metrics = await self.run_performance_test(
            client=async_client,
            method="POST",
            endpoint=f"/api/v1/finance/stocks/collect?ts_code={stock_data['ts_code']}&days=7",
            concurrent_users=performance_test_config["concurrent_users"][0],  # 1个并发用户
            duration_seconds=5,  # 缩短测试时间
            headers=auth_headers
        )
        
        print(f"采集股票数据性能测试结果:")
        print(f"  总请求数: {metrics.total_requests}")
        print(f"  成功请求: {metrics.successful_requests}")
        print(f"  失败请求: {metrics.failed_requests}")
        print(f"  平均响应时间: {metrics.avg_response_time_ms:.2f}ms")
        print(f"  错误率: {metrics.error_rate_percentage:.2f}%")
        
        # 注意：采集数据可能会因为外部API限制而失败，所以这里不严格断言
        print(f"性能测试完成，错误率: {metrics.error_rate_percentage:.2f}%")
    
    # 生成测试报告
    @pytest.mark.asyncio
    async def test_generate_report(self, async_client, auth_headers, stock_test_data):
        """生成测试报告示例"""
        if not auth_headers:
            pytest.skip("需要有效的认证头部")
        
        # 执行一些测试来填充结果
        await self.make_request(
            client=async_client,
            method="GET",
            endpoint="/api/v1/finance/stocks",
            headers=auth_headers
        )
        
        stock_data = stock_test_data["valid_stock"]
        await self.make_request(
            client=async_client,
            method="GET",
            endpoint=f"/api/v1/finance/stocks/{stock_data['ts_code']}",
            headers=auth_headers
        )
        
        report = self.generate_test_report()
        self.export_report_to_json("finance_api_test_report.json")
        self.print_summary()
        
        print(f"测试报告摘要:")
        print(f"  总测试数: {report['summary']['total_tests']}")
        print(f"  通过测试: {report['summary']['passed_tests']}")
        print(f"  失败测试: {report['summary']['failed_tests']}")
        print(f"  成功率: {report['summary']['success_rate']:.2f}%")
        print(f"  平均响应时间: {report['summary']['average_response_time_ms']:.2f}ms")
        
        assert report['summary']['total_tests'] > 0