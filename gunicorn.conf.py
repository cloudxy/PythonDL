# Gunicorn 配置文件

import multiprocessing
import os

# 服务器绑定
bind = f"127.0.0.1:8009"

# 工作进程数
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# 进程命名
proc_name = "pythondl"

# 超时设置
timeout = 120
keepalive = 5

# 日志配置
accesslog = "logs/gunicorn/access.log"
errorlog = "logs/gunicorn/error.log"
loglevel = "info"

# 访问日志格式
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程 ID 文件
pidfile = "runtimes/gunicorn.pid"

# 守护进程
daemon = False

# 工作目录
chdir = os.path.dirname(os.path.abspath(__file__))

# 最大请求数
max_requests = 1000
max_requests_jitter = 50

# 临时目录
tmp_upload_dir = "/tmp"

# 用户和组
# user = "www-data"
# group = "www-data"
