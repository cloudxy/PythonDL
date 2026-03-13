#!/bin/bash

# PythonDL 项目初始化脚本

echo "======================================"
echo "PythonDL 项目初始化"
echo "======================================"

# 创建必要的目录
echo "创建目录结构..."
mkdir -p runtimes/cache runtimes/sessions runtimes/uploads
mkdir -p temps
mkdir -p data/{finance,weather,fortune,consumption,exports}
mkdir -p files/{exports,images,videos,uploads}
mkdir -p logs/{app,error,gunicorn}
mkdir -p docs

# 创建.gitignore 文件
echo "创建.gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 环境变量
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 日志文件
logs/*
!logs/.gitkeep

# 缓存文件
runtimes/*
!runtimes/.gitkeep

# 临时文件
temps/*
!temps/.gitkeep

# 数据文件
data/*
!data/.gitkeep

# 导出文件
files/*
!files/.gitkeep

# 数据库
*.db
*.sqlite
*.sqlite3

# OS
.DS_Store
Thumbs.db
EOF

# 创建 README.md
echo "创建 README.md..."
cat > README.md << 'EOF'
# PythonDL - 全栈智能分析平台

一个集成了系统管理、金融分析、气象分析、看相算命、消费分析和爬虫采集的全栈智能分析平台。

## 技术栈

- **后端**: FastAPI + SQLAlchemy + MySQL
- **前端**: Vue 3 + Vite + Tailwind CSS
- **数据库**: MySQL 5.7+
- **缓存**: Redis / 文件缓存
- **配置**: Dynaconf
- **Web 服务器**: Gunicorn

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并修改配置：

```bash
cp .env.example .env
```

### 3. 初始化数据库

```bash
alembic upgrade head
```

### 4. 启动服务

#### 开发环境
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8009
```

#### 生产环境
```bash
gunicorn app:app -c gunicorn.conf.py
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
```

## 项目结构

详见 [docs/architecture.md](docs/architecture.md)

## API 文档

启动服务后访问：
- Swagger UI: http://localhost:8009/docs
- ReDoc: http://localhost:8009/redoc

## 功能模块

### 系统管理
- 用户管理
- 角色管理
- 权限管理
- 系统配置
- 日志管理
- 仪表盘

### 金融分析
- 股票管理
- 股票预测
- 风险评估

### 气象分析
- 气象数据管理
- 气象预测

### 看相算命
- 风水数据
- 面相数据
- 八字数据
- 周易数据
- 星座数据
- 运势数据

### 消费分析
- GDP 数据
- 人口数据
- 经济指标
- 小区数据
- 消费预测

### 爬虫采集
- 股票数据采集
- 气象数据采集
- 金融数据采集
- 算命数据采集

## 配置说明

详见 [docs/configuration.md](docs/configuration.md)

## 开发规范

详见 [docs/development.md](docs/development.md)

## 部署说明

详见 [docs/deployment.md](docs/deployment.md)

## License

MIT
EOF

# 创建.env.example
echo "创建.env.example..."
cat > .env.example << 'EOF'
# 数据库配置
DATABASE_URL=mysql+pymysql://python:123456@127.0.0.1:3306/py_demo
DATABASE_ECHO=false

# Redis 配置
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=123456
REDIS_DB=0
REDIS_URL=redis://:123456@127.0.0.1:6379/0

# 安全配置
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# 应用配置
DEBUG=true
ENVIRONMENT=development
APP_NAME=PythonDL
APP_VERSION=1.0.0

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs

# 服务器配置
HOST=127.0.0.1
PORT=8009
EOF

# 创建目录的.gitkeep 文件
echo "创建.gitkeep 文件..."
touch runtimes/.gitkeep
touch temps/.gitkeep
touch data/.gitkeep
touch files/.gitkeep
touch logs/.gitkeep

echo "======================================"
echo "初始化完成！"
echo "======================================"
echo ""
echo "下一步："
echo "1. 修改 .env 文件中的配置"
echo "2. 运行 'uv sync' 安装依赖"
echo "3. 运行 'alembic upgrade head' 初始化数据库"
echo "4. 运行 'uvicorn app:app --reload' 启动服务"
echo ""
