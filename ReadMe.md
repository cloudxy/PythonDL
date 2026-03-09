# PythonDL 项目

<p align="center">
  <img src="docs/images/logo.png" alt="PythonDL Logo" width="200">
</p>

<p align="center">
  <strong>智能分析平台 - 集金融、气象、看相算命、消费分析于一体</strong>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#技术栈">技术栈</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#项目架构">项目架构</a> •
  <a href="#部署">部署</a> •
  <a href="#贡献">贡献</a>
</p>

---

## 项目简介

PythonDL 是一个集成了多种功能模块的全栈智能分析平台，采用现代化的技术栈构建，提供数据采集、分析、预测等功能。平台包含系统管理、金融分析、气象分析、看相算命分析、消费分析和爬虫采集等核心模块。

## 功能特性

### 🎯 系统管理模块
- **用户管理** - 用户增删改查、角色分配、状态管理
- **角色管理** - 角色增删改查、权限分配
- **权限管理** - 权限增删改查、树形结构
- **系统配置** - 系统名称、样式、颜色配置
- **日志管理** - 操作日志查询、过滤
- **仪表盘** - 系统整体情况展示

### 📈 金融分析模块
- **股票管理** - 股票基础信息、行情数据管理
- **股票预测** - 基于LSTM、XGBoost等模型的股价预测
- **风险评估** - 波动率、VaR、夏普比率、最大回撤计算

### 🌤️ 气象分析模块
- **气象管理** - 气象站点、气象数据管理
- **气象预测** - 基于历史数据的天气预测

### 🔮 看相算命分析模块
- **数据管理** - 风水、面相、八字、周易、星座、运势数据
- **分析功能** - 综合分析、专业解读、个性化建议

### 📊 消费分析模块
- **数据管理** - GDP、人口、经济指标、小区数据
- **消费预测** - 宏观消费趋势预测、政策建议

### 🕷️ 爬虫采集模块
- **数据采集** - 金融、气象、看相算命、宏观消费数据采集
- **后台任务** - 异步采集、定时采集

## 技术栈

### 后端
| 技术 | 版本 | 描述 |
|------|------|------|
| Python | 3.13+ | 主要开发语言 |
| FastAPI | 0.100+ | 现代高性能Web框架 |
| SQLAlchemy | 2.0+ | ORM框架 |
| Alembic | 1.12+ | 数据库迁移工具 |
| Dynaconf | 3.2+ | 配置管理 |
| Gunicorn | 21.2+ | WSGI服务器 |
| MySQL | 5.7+ | 主数据库 |
| Redis | 6.0+ | 缓存数据库（可选） |

### 前端
| 技术 | 版本 | 描述 |
|------|------|------|
| Vue | 3.x | 前端框架 |
| Vite | 5.x | 构建工具 |
| Tailwind CSS | 3.x | 样式框架 |
| Vue Router | 4.x | 路由管理 |
| Pinia | 2.x | 状态管理 |

### AI/ML
| 技术 | 版本 | 描述 |
|------|------|------|
| TensorFlow | 2.20+ | 深度学习框架 |
| XGBoost | 3.1+ | 梯度提升框架 |
| Scikit-learn | 1.8+ | 机器学习库 |
| NumPy | 2.3+ | 数值计算 |
| Pandas | 2.3+ | 数据分析 |

## 快速开始

### 环境要求
- Python 3.13+
- MySQL 5.7+
- Redis 6.0+（可选）
- UV包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/pythondl.git
cd pythondl
```

2. **安装依赖**
```bash
# 安装UV
pip install uv

# 安装项目依赖
uv sync
```

3. **配置环境变量**
```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑配置文件
vim .env
```

4. **创建数据库**
```sql
CREATE DATABASE py_demo CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

5. **运行数据库迁移**
```bash
alembic upgrade head
```

6. **启动开发服务器**
```bash
# 开发环境
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 生产环境
uv run gunicorn -c gunicorn.conf.py app:app
```

7. **访问应用**
- API文档: http://localhost:8000/docs
- 登录页面: http://localhost:8000/static/pages/auth/login.html
- 健康检查: http://localhost:8000/health

## 项目架构

```
PythonDL/
├── alembic/             # 数据库迁移
├── app/                 # 应用主目录
│   ├── admin/           # 系统管理模块
│   ├── api/             # API接口
│   ├── core/            # 核心功能
│   ├── models/          # 数据模型
│   ├── schemas/         # 数据验证
│   ├── services/        # 业务逻辑
│   └── static/          # 静态资源
├── data/                # 数据文件
├── docs/                # 项目文档
├── files/               # 导出文件
├── frontend/            # 前端代码
├── logs/                # 日志文件
├── runtimes/            # 缓存信息
├── temps/               # 临时文件
├── tests/               # 测试代码
└── utils/               # 工具函数
```

详细架构文档请参考 [architecture.md](docs/architecture.md)

## 配置说明

### 主要配置文件
- `settings.toml` - 主要配置文件
- `.env` - 环境变量配置
- `gunicorn.conf.py` - Gunicorn配置

### 数据库配置
```toml
[database]
host = "127.0.0.1"
port = 3306
user = "python"
password = "123456"
name = "py_demo"
```

### Redis配置
```toml
[redis]
host = "127.0.0.1"
port = 6379
password = "123456"
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t pythondl .

# 运行容器
docker run -d -p 8000:8000 pythondl
```

### Docker Compose部署

```bash
docker-compose up -d
```

### 生产环境配置

1. 配置环境变量
2. 设置数据库连接
3. 配置Redis（可选）
4. 设置密钥
5. 配置日志
6. 启动服务

## 开发指南

### 代码规范
- 使用Black进行代码格式化
- 使用Flake8进行代码检查
- 使用MyPy进行类型检查
- 遵循PEP 8规范

### 运行测试

```bash
# 运行测试
uv run pytest

# 运行测试并生成覆盖率
uv run pytest --cov=app --cov-report=html
```

### 代码检查

```bash
# 格式化代码
uv run black app/

# 代码检查
uv run flake8 app/

# 类型检查
uv run mypy app/
```

## 日志系统

- 日志存储在 `logs` 目录下
- 按天流转，区分普通日志和错误日志
- 支持自动压缩和清理

## 缓存系统

- 使用文件缓存，存储在 `runtimes/cache` 目录
- 支持 Redis 缓存（可选）
- 支持缓存预热和过期管理

## 安全措施

- 密码加密存储
- 基于角色的权限控制
- 速率限制
- 输入验证
- CORS 配置
- SQL注入防护
- XSS防护

## 性能优化

- 缓存策略
- 数据库索引
- 异步处理
- 连接池
- 代码优化

## API文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 贡献

欢迎提交问题和拉取请求！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目地址: https://github.com/yourusername/pythondl
- 问题反馈: https://github.com/yourusername/pythondl/issues

---

<p align="center">
  Made with ❤️ by PythonDL Team
</p>
