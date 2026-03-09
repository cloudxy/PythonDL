"""API测试基础类

此模块包含API测试的基础类，提供通用的测试方法和断言，支持功能测试、性能测试、安全测试和契约测试。
"""
import pytest
import httpx
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import statistics

@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    percentile_50_ms: float
    percentile_90_ms: float
    percentile_95_ms: float
    percentile_99_ms: float
    requests_per_second: float
    error_rate_percentage: float

@dataclass
class SecurityTestResult:
    """安全测试结果数据类"""
    test_name: str
    endpoint: str
    method: str
    payload: str
    status_code: int
    success: bool
    error_message: Optional[str] = None

class BaseAPITest:
    """API测试基础类
    
    提供通用的测试方法和断言，所有API测试类都应继承此类。
    """
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}  
        self.security_test_results: List[SecurityTestResult] = []
        self.contract_test_results: List[Dict] = []
        self.test_summary: Dict[str, Any] = defaultdict(int)
    
    async def make_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        endpoint: str,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None
    ) -> httpx.Response:
        """发送HTTP请求
        
        Args:
            client: HTTP客户端
            method: HTTP方法 (GET, POST, PUT, DELETE)
            endpoint: API端点
            headers: 请求头
            json_data: JSON请求体
            params: 查询参数
            files: 文件上传
            
        Returns:
            HTTP响应对象
        """
        start_time = time.time()
        response = None
        
        try:
            if method.upper() == "GET":
                response = await client.get(endpoint, headers=headers, params=params)
            elif method.upper() == "POST":
                if files:
                    response = await client.post(endpoint, headers=headers, data=json_data, files=files)
                else:
                    response = await client.post(endpoint, headers=headers, json=json_data, params=params)
            elif method.upper() == "PUT":
                response = await client.put(endpoint, headers=headers, json=json_data, params=params)
            elif method.upper() == "DELETE":
                response = await client.delete(endpoint, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response_time_ms = (time.time() - start_time) * 1000
            
            # 记录测试结果
            test_result = TestResult(
                test_name=self._get_test_name(),
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                success=response.status_code < 400
            )
            self.test_results.append(test_result)
            
            # 更新测试摘要
            self.test_summary["total"] += 1
            if test_result.success:
                self.test_summary["passed"] += 1
            else:
                self.test_summary["failed"] += 1
            
            return response
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            test_result = TestResult(
                test_name=self._get_test_name(),
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e)
            )
            self.test_results.append(test_result)
            
            # 更新测试摘要
            self.test_summary["total"] += 1
            self.test_summary["failed"] += 1
            
            raise
    
    def _get_test_name(self) -> str:
        """获取当前测试名称"""
        # 使用pytest的当前测试项名称
        import inspect
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name.startswith('test_'):
                return frame.f_code.co_name
            frame = frame.f_back
        return "unknown_test"
    
    # 断言方法
    def assert_status_code(self, response: httpx.Response, expected_code: int):
        """断言状态码"""
        assert response.status_code == expected_code, \
            f"Expected status code {expected_code}, got {response.status_code}"
    
    def assert_response_schema(self, response: httpx.Response, expected_keys: List[str]):
        """断言响应包含预期的键"""
        data = response.json()
        if isinstance(data, dict):
            for key in expected_keys:
                assert key in data, f"Expected key '{key}' not found in response"
        elif isinstance(data, list) and len(data) > 0:
            for key in expected_keys:
                assert key in data[0], f"Expected key '{key}' not found in first item of response array"
    
    def assert_response_time(self, response_time_ms: float, max_threshold_ms: float):
        """断言响应时间"""
        assert response_time_ms <= max_threshold_ms, \
            f"Response time {response_time_ms:.2f}ms exceeds threshold {max_threshold_ms}ms"
    
    def assert_error_message(self, response: httpx.Response, expected_message: str = None):
        """断言错误消息"""
        data = response.json()
        if expected_message:
            if 'message' in data:
                assert expected_message in data['message'], \
                    f"Expected error message containing '{expected_message}', got '{data.get('message')}'"
            elif 'error' in data:
                assert expected_message in data['error'], \
                    f"Expected error message containing '{expected_message}', got '{data.get('error')}'"
    
    def assert_pagination(self, response: httpx.Response):
        """断言分页响应结构"""
        data = response.json()
        if 'data' in data and 'total' in data and 'skip' in data and 'limit' in data:
            assert isinstance(data['data'], list), "Pagination data should be a list"
            assert isinstance(data['total'], int), "Total should be an integer"
            assert isinstance(data['skip'], int), "Skip should be an integer"
            assert isinstance(data['limit'], int), "Limit should be an integer"
    
    # 性能测试方法
    async def run_performance_test(
        self,
        client: httpx.AsyncClient,
        method: str,
        endpoint: str,
        concurrent_users: int,
        duration_seconds: int,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> PerformanceMetrics:
        """运行性能测试
        
        Args:
            client: HTTP客户端
            method: HTTP方法
            endpoint: API端点
            concurrent_users: 并发用户数
            duration_seconds: 测试持续时间（秒）
            headers: 请求头
            json_data: 请求体
            
        Returns:
            性能指标
        """
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # 创建并发任务
        async def make_single_request():
            nonlocal successful_requests, failed_requests
            request_start = time.time()
            try:
                if method.upper() == "GET":
                    response = await client.get(endpoint, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(endpoint, headers=headers, json=json_data)
                else:
                    raise ValueError(f"Unsupported method for performance test: {method}")
                
                response_time = (time.time() - request_start) * 1000
                response_times.append(response_time)
                
                if response.status_code < 400:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    
            except Exception:
                failed_requests += 1
        
        # 运行测试
        while time.time() - start_time < duration_seconds:
            tasks = []
            for _ in range(concurrent_users):
                tasks.append(make_single_request())
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)  # 避免过于密集的请求
        
        # 计算性能指标
        total_requests = successful_requests + failed_requests
        if total_requests == 0:
            return PerformanceMetrics(
                endpoint=endpoint,
                method=method,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                percentile_50_ms=0,
                percentile_90_ms=0,
                percentile_95_ms=0,
                percentile_99_ms=0,
                requests_per_second=0,
                error_rate_percentage=0
            )
        
        response_times_sorted = sorted(response_times)
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        def percentile(p):
            if not response_times_sorted:
                return 0
            index = int(p / 100 * len(response_times_sorted))
            return response_times_sorted[min(index, len(response_times_sorted) - 1)]
        
        metrics = PerformanceMetrics(
            endpoint=endpoint,
            method=method,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            percentile_50_ms=percentile(50),
            percentile_90_ms=percentile(90),
            percentile_95_ms=percentile(95),
            percentile_99_ms=percentile(99),
            requests_per_second=total_requests / duration_seconds,
            error_rate_percentage=(failed_requests / total_requests) * 100
        )
        
        self.performance_metrics[f"{method}_{endpoint}"] = metrics
        return metrics
    
    # 安全测试方法
    async def test_sql_injection(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        payloads: List[str],
        headers: Optional[Dict] = None
    ) -> List[SecurityTestResult]:
        """测试SQL注入漏洞
        
        Args:
            client: HTTP客户端
            endpoint: API端点
            method: HTTP方法
            payloads: SQL注入payload列表
            headers: 请求头
            
        Returns:
            安全测试结果列表
        """
        results = []
        
        for payload in payloads:
            # 根据端点类型构造测试数据
            test_data = self._create_injection_test_data(endpoint, payload)
            
            try:
                if method.upper() == "GET":
                    response = await client.get(endpoint, params=test_data, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(endpoint, json=test_data, headers=headers)
                else:
                    continue
                
                success = not self._is_sql_injection_indicated(response)
                error_message = f"Payload: {payload}" if not success else None
                
                result = SecurityTestResult(
                    test_name="sql_injection_test",
                    endpoint=endpoint,
                    method=method,
                    payload=payload,
                    status_code=response.status_code,
                    success=success,
                    error_message=error_message
                )
                results.append(result)
                self.security_test_results.append(result)
                
            except Exception as e:
                result = SecurityTestResult(
                    test_name="sql_injection_test",
                    endpoint=endpoint,
                    method=method,
                    payload=payload,
                    status_code=0,
                    success=False,
                    error_message=f"Exception with payload {payload}: {str(e)}"
                )
                results.append(result)
                self.security_test_results.append(result)
        
        return results
    
    async def test_xss_injection(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        payloads: List[str],
        headers: Optional[Dict] = None
    ) -> List[SecurityTestResult]:
        """测试XSS注入漏洞
        
        Args:
            client: HTTP客户端
            endpoint: API端点
            method: HTTP方法
            payloads: XSS注入payload列表
            headers: 请求头
            
        Returns:
            安全测试结果列表
        """
        results = []
        
        for payload in payloads:
            test_data = self._create_injection_test_data(endpoint, payload)
            
            try:
                if method.upper() == "POST":
                    response = await client.post(endpoint, json=test_data, headers=headers)
                else:
                    continue
                
                success = not self._is_xss_indicated(response)
                error_message = f"Payload: {payload}" if not success else None
                
                result = SecurityTestResult(
                    test_name="xss_injection_test",
                    endpoint=endpoint,
                    method=method,
                    payload=payload,
                    status_code=response.status_code,
                    success=success,
                    error_message=error_message
                )
                results.append(result)
                self.security_test_results.append(result)
                
            except Exception as e:
                result = SecurityTestResult(
                    test_name="xss_injection_test",
                    endpoint=endpoint,
                    method=method,
                    payload=payload,
                    status_code=0,
                    success=False,
                    error_message=f"Exception with payload {payload}: {str(e)}"
                )
                results.append(result)
                self.security_test_results.append(result)
        
        return results
    
    # 契约测试方法
    async def test_contract_validation(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        expected_schema: Dict,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """测试API契约验证
        
        Args:
            client: HTTP客户端
            endpoint: API端点
            method: HTTP方法
            expected_schema: 预期的响应schema
            headers: 请求头
            json_data: 请求数据
            
        Returns:
            契约测试结果
        """
        try:
            if method.upper() == "GET":
                response = await client.get(endpoint, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(endpoint, json=json_data, headers=headers)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}
            
            if response.status_code < 400:
                response_data = response.json()
                # 简单的结构验证
                success = self._validate_response_structure(response_data, expected_schema)
                if success:
                    result = {"success": True, "endpoint": endpoint, "method": method}
                else:
                    result = {"success": False, "endpoint": endpoint, "method": method, "error": "Response structure validation failed"}
            else:
                result = {"success": True, "endpoint": endpoint, "method": method, "note": f"Received error status: {response.status_code}"}
            
            self.contract_test_results.append(result)
            return result
            
        except Exception as e:
            result = {"success": False, "endpoint": endpoint, "method": method, "error": str(e)}
            self.contract_test_results.append(result)
            return result
    
    def _validate_response_structure(self, response_data: Dict, expected_schema: Dict) -> bool:
        """验证响应结构是否符合预期
        
        Args:
            response_data: 响应数据
            expected_schema: 预期的schema结构
            
        Returns:
            是否验证成功
        """
        try:
            if "required" in expected_schema:
                for field in expected_schema["required"]:
                    if field not in response_data:
                        return False
            
            if "properties" in expected_schema:
                for field, prop_schema in expected_schema["properties"].items():
                    if field in response_data:
                        field_value = response_data[field]
                        if "type" in prop_schema:
                            expected_type = prop_schema["type"]
                            if expected_type == "string" and not isinstance(field_value, str):
                                return False
                            elif expected_type == "number" and not isinstance(field_value, (int, float)):
                                return False
                            elif expected_type == "integer" and not isinstance(field_value, int):
                                return False
                            elif expected_type == "boolean" and not isinstance(field_value, bool):
                                return False
                            elif expected_type == "array" and not isinstance(field_value, list):
                                return False
                            elif expected_type == "object" and not isinstance(field_value, dict):
                                return False
        except Exception:
            return False
        
        return True
    
    def _create_injection_test_data(self, endpoint: str, payload: str) -> Dict:
        """根据端点创建注入测试数据"""
        # 根据端点类型返回不同的测试数据
        if "user" in endpoint.lower():
            return {"username": payload, "password": "test123"}
        elif "stock" in endpoint.lower():
            return {"ts_code": payload, "days": 30}
        elif "weather" in endpoint.lower():
            return {"station_code": payload, "days": 7}
        else:
            return {"input": payload}
    
    def _is_sql_injection_indicated(self, response: httpx.Response) -> bool:
        """检查响应是否表明存在SQL注入漏洞"""
        if response.status_code >= 500:
            return True
        
        response_text = response.text.lower()
        sql_error_indicators = [
            "sql", "syntax", "database", "mysql", "postgresql",
            "oracle", "sqlite", "integrity constraint", "exception", "error"
        ]
        
        for indicator in sql_error_indicators:
            if indicator in response_text:
                return True
        
        return False
    
    def _is_xss_indicated(self, response: httpx.Response) -> bool:
        """检查响应是否表明存在XSS漏洞"""
        response_text = response.text.lower()
        xss_indicators = [
            "<script>", "javascript:", "onerror=", "onload=",
            "<iframe>", "<img src=x", "alert("
        ]
        
        for indicator in xss_indicators:
            if indicator in response_text:
                return True
        
        return False
    
    # 报告生成方法
    def generate_test_report(self) -> Dict:
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(r.response_time_ms for r in self.test_results) / total_tests if total_tests > 0 else 0
        
        # 安全测试统计
        total_security_tests = len(self.security_test_results)
        passed_security_tests = sum(1 for r in self.security_test_results if r.success)
        failed_security_tests = total_security_tests - passed_security_tests
        
        # 契约测试统计
        total_contract_tests = len(self.contract_test_results)
        passed_contract_tests = sum(1 for r in self.contract_test_results if r.get('success', False))
        failed_contract_tests = total_contract_tests - passed_contract_tests
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "average_response_time_ms": avg_response_time,
                "security_tests": {
                    "total": total_security_tests,
                    "passed": passed_security_tests,
                    "failed": failed_security_tests,
                    "success_rate": (passed_security_tests / total_security_tests * 100) if total_security_tests > 0 else 0
                },
                "contract_tests": {
                    "total": total_contract_tests,
                    "passed": passed_contract_tests,
                    "failed": failed_contract_tests,
                    "success_rate": (passed_contract_tests / total_contract_tests * 100) if total_contract_tests > 0 else 0
                }
            },
            "test_results": [r.__dict__ for r in self.test_results],
            "security_test_results": [r.__dict__ for r in self.security_test_results],
            "contract_test_results": self.contract_test_results,
            "performance_metrics": {k: m.__dict__ for k, m in self.performance_metrics.items()},
            "generated_at": datetime.now().isoformat()
        }
    
    def export_report_to_json(self, filename: str) -> None:
        """导出测试报告到JSON文件
        
        Args:
            filename: 输出文件名
        """
        report = self.generate_test_report()
        
        # 递归转换datetime对象为字符串
        def serialize_datetime(obj):
            if isinstance(obj, dict):
                return {key: serialize_datetime(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetime(item) for item in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj
        
        serialized_report = serialize_datetime(report)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serialized_report, f, ensure_ascii=False, indent=2)
    
    def print_summary(self) -> None:
        """打印测试摘要"""
        report = self.generate_test_report()
        summary = report['summary']
        
        print("=======================================")
        print("API测试报告摘要")
        print("=======================================")
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"成功率: {summary['success_rate']:.2f}%")
        print(f"平均响应时间: {summary['average_response_time_ms']:.2f}ms")
        print()
        print("安全测试:")
        print(f"  总测试: {summary['security_tests']['total']}")
        print(f"  通过: {summary['security_tests']['passed']}")
        print(f"  失败: {summary['security_tests']['failed']}")
        print(f"  成功率: {summary['security_tests']['success_rate']:.2f}%")
        print()
        print("契约测试:")
        print(f"  总测试: {summary['contract_tests']['total']}")
        print(f"  通过: {summary['contract_tests']['passed']}")
        print(f"  失败: {summary['contract_tests']['failed']}")
        print(f"  成功率: {summary['contract_tests']['success_rate']:.2f}%")
        print("=======================================")