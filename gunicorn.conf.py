# Gunicorn配置文件

# 绑定地址和端口
bind = '0.0.0.0:8000'

# 工作进程数
workers = 4

# 工作进程类型
worker_class = 'uvicorn.workers.UvicornWorker'

# 工作进程超时时间
timeout = 120

# 最大请求数
timeout = 120

# 最大请求数
max_requests = 10000
max_requests_jitter = 1000

# 日志配置
accesslog = './logs/gunicorn/access.log'
errorlog = './logs/gunicorn/error.log'
loglevel = 'info'

# 进程名称
proc_name = 'PythonDL'

# 启动时的环境变量
env = {
    'PYTHONDL_ENV': 'production'
}

# 预加载应用
preload_app = True

# 后台运行
daemon = False

# 重启信号
graceful_timeout = 30
