"""FastAPI应用初始化

此模块负责FastAPI应用的初始化和配置。
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, RedirectResponse
from pathlib import Path
from app.core.config import config
from app.core.logger import get_logger

logger = get_logger("app")

# 创建FastAPI应用实例
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    logger.info("Application startup")
    logger.info(f"API documentation available at {config.API_DOCS_URL}")
    logger.info(f"Static files mounted at /static")
    yield
    # 关闭事件
    logger.info("Application shutdown")

app = FastAPI(
    title="PythonDL API",
    description="PythonDL 高级算法框架 API",
    version=config.APP_VERSION,
    docs_url=config.API_DOCS_URL,
    redoc_url=config.API_REDOC_URL,
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

# 添加速率限制中间件
from app.core.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, max_requests=1000, window_seconds=60)

# 添加请求上下文中间件
from app.core.monitoring import RequestContextMiddleware
app.add_middleware(RequestContextMiddleware)

# 添加性能监控中间件
from app.core.monitoring import PerformanceMonitoringMiddleware
app.add_middleware(PerformanceMonitoringMiddleware)

# 添加日志记录中间件
from app.core.middleware.logging import LoggingMiddleware
app.add_middleware(LoggingMiddleware)

# 配置静态文件服务
static_dir = Path(__file__).parent / "static"

# 自定义静态文件服务，支持压缩文件和缓存控制
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

class CustomStaticFiles(StaticFiles):
    """自定义静态文件服务，支持压缩文件和缓存控制"""
    
    async def get_response(self, path: str, scope):
        # 尝试提供压缩文件
        gzip_path = f"{path}.gz"
        if os.path.exists(os.path.join(self.directory, gzip_path)):
            # 检查客户端是否支持gzip
            accept_encoding = scope.get("headers", {}).get(b"accept-encoding", b"")
            if b"gzip" in accept_encoding:
                response = await super().get_response(gzip_path, scope)
                response.headers["Content-Encoding"] = "gzip"
                response.headers["Content-Length"] = str(os.path.getsize(os.path.join(self.directory, gzip_path)))
                # 添加缓存控制头
                response.headers["Cache-Control"] = "public, max-age=86400"  # 24小时缓存
                return response
        
        # 提供原始文件
        response = await super().get_response(path, scope)
        # 添加缓存控制头
        response.headers["Cache-Control"] = "public, max-age=86400"  # 24小时缓存
        return response

# 挂载静态文件服务
app.mount("/static", CustomStaticFiles(directory=str(static_dir)), name="static")

# 导入路由
from app.api import v1_router
# 导入异常处理器
from app.core.exceptions import setup_exception_handlers

# 设置异常处理器
setup_exception_handlers(app)

# 注册路由
app.include_router(v1_router)

# 根路径
@app.get("/")
def read_root():
    """API根路径"""
    return {
        "message": "Welcome to PythonDL API",
        "version": config.APP_VERSION,
        "docs": config.API_DOCS_URL,
        "admin": "/static/pages/auth/login.html"
    }

# 重定向旧登录路径到新路径
@app.get("/static/pages/login.html")
def redirect_old_login_path():
    """将旧登录路径重定向到新路径"""
    return RedirectResponse(url="/static/pages/auth/login.html")

# 健康检查端点
@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

# 处理@vite/client请求，避免404错误
@app.get("/@vite/client")
def handle_vite_client():
    """处理Vite客户端请求，避免404并防止内存错误"""
    try:
        # 简化响应内容，减少内存分配
        vite_client_content = "// Vite client placeholder (fixed for memory safety)"
        # 显式构造响应，确保资源正常释放
        response = Response(
            content=vite_client_content,
            media_type="application/javascript",
            status_code=200
        )
        logger.info("Successfully handled /@vite/client request")
        return response
    except Exception as e:
        # 捕获所有异常，避免未处理异常导致的内存释放异常
        logger.error(f"Error handling /@vite/client: {str(e)}", exc_info=True)
        # 异常时返回兜底响应，确保进程不崩溃
        return Response(
            content="// Vite client fallback (error occurred)",
            media_type="application/javascript",
            status_code=200
        )

