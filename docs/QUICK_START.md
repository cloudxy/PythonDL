# PythonDL 快速启动指南

## 🚀 5 分钟快速启动

### 前置要求

确保已安装以下软件：
- ✅ Python 3.13+
- ✅ Node.js 18+
- ✅ MySQL 8.0+
- ✅ Redis 5.0+

### 第一步：安装依赖

```bash
# 安装后端依赖
cd /Users/xuyun/Projects/python_projects/PythonDL
uv sync

# 安装前端依赖
cd frontend
npm install
```

### 第二步：配置环境

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑 .env 文件，配置以下关键参数：
# - 数据库连接
# - Redis 连接
# - SMTP 邮箱
# - 密钥配置
```

**最小化配置示例** (.env):
```ini
# 应用配置
APP_DEBUG=true
SECRET_KEY=your-secret-key-change-in-production

# 数据库配置
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=python
DB_PASSWORD=123456
DB_NAME=py_demo

# Redis 配置
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=123456

# SMTP 邮箱配置 (可选)
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USER=your_email@qq.com
SMTP_PASSWORD=your_auth_code
```

### 第三步：启动数据库和 Redis

```bash
# macOS (使用 Homebrew)
brew services start mysql
brew services start redis

# Linux (systemctl)
sudo systemctl start mysql
sudo systemctl start redis

# Windows (服务管理器)
# 在服务管理器中启动 MySQL 和 Redis 服务
```

### 第四步：数据库迁移

```bash
# 执行数据库迁移，创建所有表
alembic upgrade head
```

### 第五步：启动后端服务

```bash
# 开发模式 (自动重载)
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 或使用 Gunicorn (生产模式)
uv run gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

**访问 API 文档**: http://localhost:8000/docs

### 第六步：启动前端服务

```bash
cd frontend

# 开发模式
npm run dev

# 访问前端
# http://localhost:3000
```

---

## ✅ 验证启动成功

### 检查后端

```bash
# 健康检查
curl http://localhost:8000/health

# 预期输出：
# {"status":"healthy"}
```

### 检查 API 文档

打开浏览器访问：http://localhost:8000/docs

### 检查前端

打开浏览器访问：http://localhost:3000

---

## 🔧 常见问题

### 1. 数据库连接失败

**错误**: `Can't connect to MySQL server`

**解决**:
```bash
# 检查 MySQL 是否运行
brew services list | grep mysql

# 重启 MySQL
brew services restart mysql

# 检查连接配置
cat .env | grep DB_
```

### 2. Redis 连接失败

**错误**: `Error connecting to Redis`

**解决**:
```bash
# 检查 Redis 是否运行
brew services list | grep redis

# 重启 Redis
brew services restart redis

# 测试 Redis 连接
redis-cli ping
# 预期输出：PONG
```

### 3. 端口被占用

**错误**: `Address already in use`

**解决**:
```bash
# 查找占用端口的进程
lsof -i :8000
lsof -i :3000

# 杀死进程
kill -9 <PID>

# 或修改配置使用其他端口
```

### 4. 模块导入错误

**错误**: `ModuleNotFoundError`

**解决**:
```bash
# 重新安装依赖
uv sync --reinstall

# 检查 Python 路径
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 5. 前端依赖错误

**错误**: `npm install` 失败

**解决**:
```bash
cd frontend

# 删除 node_modules 和 package-lock.json
rm -rf node_modules package-lock.json

# 重新安装
npm install
```

---

## 📊 系统状态检查

运行综合分析脚本：

```bash
uv run python tests/comprehensive_system_analysis.py
```

**预期输出**: 总体完整度 100% ✅

---

## 🎯 下一步

### 1. 创建测试数据

```bash
# 运行测试脚本创建示例数据
# (如果有初始化脚本的话)
```

### 2. 测试 API 接口

访问 http://localhost:8000/docs 测试各个 API 接口

### 3. 登录系统

使用默认管理员账号登录前端系统
(如果有默认账号的话)

### 4. 配置爬虫任务

在系统管理界面配置定时爬虫任务

---

## 📚 文档链接

- [完整分析报告](COMPREHENSIVE_ANALYSIS_REPORT.md)
- [新功能完成报告](NEW_FEATURES_COMPLETE.md)
- [项目架构文档](docs/architecture.md)
- [API 参考文档](API_REFERENCE.md)

---

## 🆘 获取帮助

### 日志查看

```bash
# 查看后端日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/app_error.log
```

### 调试模式

在 `.env` 中设置：
```ini
APP_DEBUG=true
LOG_LEVEL=DEBUG
```

### 数据库检查

```bash
# 连接数据库
mysql -u python -p py_demo

# 查看所有表
SHOW TABLES;
```

---

## 🎉 启动成功！

如果所有步骤都顺利完成，您现在应该：

✅ 后端服务运行在 http://localhost:8000  
✅ 前端服务运行在 http://localhost:3000  
✅ API 文档可访问 http://localhost:8000/docs  
✅ 数据库连接正常  
✅ Redis 缓存可用  

**开始使用 PythonDL 吧！** 🚀
