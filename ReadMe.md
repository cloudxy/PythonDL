# PythonDL - 全栈智能分析平台

<div align="center">

![PythonDL](https://img.shields.io/badge/PythonDL-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.13+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Vue](https://img.shields.io/badge/Vue-3.4+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**一个集成了系统管理、金融分析、气象分析、看相算命、消费分析和爬虫采集的全栈智能分析平台**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [文档](#-文档) • [API](#-api) • [部署](#-部署)

</div>

---

## 📖 项目简介

PythonDL 是一个现代化的全栈智能分析平台，采用前后端分离架构，集成了六大核心模块：系统管理、金融分析、气象分析、看相算命、消费分析和爬虫采集。平台提供完整的 CRUD 功能、AI/ML 智能预测、数据可视化分析等能力，适用于多种业务场景。

### 核心优势

- 🎯 **模块化设计**: 清晰的功能模块划分，易于扩展和维护
- 🚀 **高性能架构**: FastAPI + SQLAlchemy + MySQL，支持高并发
- 🎨 **顶级 UI 设计**: 基于 Vue 3 + TailwindCSS 的现代化界面
- 🤖 **AI/ML 集成**: TensorFlow、XGBoost 智能预测算法
- 🔐 **安全可靠**: JWT 认证、RBAC 权限、多重安全防护
- 📊 **数据可视化**: 丰富的图表组件和数据分析能力

---

## ✨ 功能特性

### 🏛️ 系统管理模块

完整的后台管理系统，支持多角色多权限控制。

| 功能 | 描述 | 状态 |
|------|------|------|
| 用户管理 | 用户 CRUD、角色分配、状态管理 | ✅ |
| 角色管理 | 角色 CRUD、权限配置 | ✅ |
| 权限管理 | 权限 CRUD、菜单权限、接口权限 | ✅ |
| 系统配置 | 系统参数、界面样式、颜色配置 | ✅ |
| 日志管理 | 操作日志、登录日志、系统日志 | ✅ |
| 仪表盘 | 系统监控、数据统计、快捷操作 | ✅ |

**特性**:
- 🔐 JWT 双令牌认证（Access Token + Refresh Token）
- 👥 RBAC 权限模型（用户 → 角色 → 权限）
- 📝 完整的操作日志记录
- 🎨 可配置的系统主题和样式

### 📈 金融分析模块

专业的股票分析和预测系统。

| 功能 | 描述 | 状态 |
|------|------|------|
| 股票管理 | 股票列表、详情、编辑、删除 | ✅ |
| 股票预测 | LSTM/XGBoost 价格预测、趋势分析 | ✅ |
| 风险评估 | 多维度风险评估、风险等级划分 | ✅ |
| 数据采集 | 自动采集股票行情数据 | ✅ |

**特性**:
- 📊 支持中国 A 股市场数据采集
- 🤖 LSTM 深度学习预测模型
- 📉 多维度风险评估指标
- 💹 实时股票行情数据展示

### 🌤️ 气象分析模块

全面的气象数据分析和预测系统。

| 功能 | 描述 | 状态 |
|------|------|------|
| 站点管理 | 气象站点 CRUD、位置管理 | ✅ |
| 数据管理 | 气象数据 CRUD、搜索、导出 | ✅ |
| 气象预测 | 温度预测、天气状况预测 | ✅ |
| 数据采集 | 自动采集全国气象数据 | ✅ |

**特性**:
- 🌡️ 支持全国气象站点数据
- 📅 近一年历史数据存储
- 🌦️ 智能天气预报算法
- 📊 气象数据可视化分析

### 🔮 看相算命模块

传统的周易八字分析系统。

| 功能 | 描述 | 状态 |
|------|------|------|
| 风水管理 | 风水数据 CRUD、方位分析 | ✅ |
| 面相管理 | 面相数据 CRUD、特征分析 | ✅ |
| 八字管理 | 八字排盘、命理分析 | ✅ |
| 周易管理 | 周易卦象、占卜分析 | ✅ |
| 星座管理 | 星座数据、性格分析 | ✅ |
| 运势管理 | 运势预测、吉凶分析 | ✅ |
| 综合分析 | 多维度综合分析、报告生成 | ✅ |

**特性**:
- 📖 传统周易理论支持
- 🔢 八字排盘算法
- ⭐ 星座运势分析
- 📊 多维度综合分析

### 💰 消费分析模块

宏观经济数据分析系统。

| 功能 | 描述 | 状态 |
|------|------|------|
| GDP 管理 | 全国/省/市 GDP 数据 CRUD | ✅ |
| 人口管理 | 人口数据 CRUD、统计分析 | ✅ |
| 经济指标 | 消费、出行、工业生产数据 | ✅ |
| 小区数据 | 社区数据 CRUD、分布分析 | ✅ |
| 消费预测 | 消费趋势预测、分析建议 | ✅ |

**特性**:
- 📊 多维度经济指标分析
- 🌆 全国/省/市三级数据
- 📈 消费趋势预测模型
- 💡 专业分析报告生成

### 🕷️ 爬虫采集模块

强大的数据采集系统。

| 功能 | 描述 | 状态 |
|------|------|------|
| 股票采集 | 股票走势、行情数据采集 | ✅ |
| 气象采集 | 天气数据、气象指标采集 | ✅ |
| 算命采集 | 风水、面相、八字等数据采集 | ✅ |
| 消费采集 | GDP、人口、经济指标采集 | ✅ |
| 任务管理 | 采集任务 CRUD、调度管理 | ✅ |

**特性**:
- 🕸️ 支持多种数据源
- ⏰ 定时任务调度
- 📥 数据清洗和验证
- 📊 采集监控和日志

---

## 🛠️ 技术栈

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **框架** | FastAPI 0.100+ | Web 框架 |
| **语言** | Python 3.13+ | 编程语言 |
| **环境管理** | UV | 包管理 |
| **配置管理** | Dynaconf 3.2+ | 配置管理 |
| **数据库** | MySQL 8.0+ | 关系数据库 |
| **ORM** | SQLAlchemy 2.0+ | 对象关系映射 |
| **迁移工具** | Alembic 1.12+ | 数据库迁移 |
| **数据验证** | Pydantic 2.0+ | 数据验证 |
| **认证授权** | python-jose 3.3+ | JWT 认证 |
| **密码加密** | bcrypt 4.0+ | 密码哈希 |
| **缓存** | 文件缓存/Redis 5.0+ | 数据缓存 |
| **日志** | logging + structlog | 日志记录 |
| **Web 服务器** | Gunicorn 21.2+ | WSGI 服务器 |
| **AI/ML** | TensorFlow 2.20+ | 深度学习 |
| **机器学习** | XGBoost 3.1+ | 机器学习 |
| **数据分析** | pandas 2.3+ | 数据处理 |
| **科学计算** | numpy 2.3+ | 数值计算 |

### 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **框架** | Vue 3.4+ | 前端框架 |
| **构建工具** | Vite 5.4+ | 构建工具 |
| **样式** | TailwindCSS 3.4+ | CSS 框架 |
| **路由** | Vue Router 4+ | 路由管理 |
| **状态管理** | Pinia | 状态管理 |
| **HTTP 客户端** | Axios | HTTP 请求 |
| **图表库** | Chart.js + vue-chartjs | 数据可视化 |
| **UI 组件** | 自研组件库 | UI 组件 |

### DevOps 工具

| 技术 | 版本 | 用途 |
|------|------|------|
| **CI/CD** | GitHub Actions | 持续集成 |
| **容器化** | Docker + Docker Compose | 容器部署 |
| **测试框架** | pytest 7.0+ | 单元测试 |
| **性能测试** | 自研框架 | 性能测试 |
| **API 测试** | Playwright | E2E 测试 |

---

## 🚀 快速开始

### 前置要求

- Python 3.13+
- Node.js 18+
- MySQL 8.0+
- UV 包管理器

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/PythonDL.git
cd PythonDL
```

### 2. 安装后端依赖

```bash
# 使用 UV 安装依赖
uv sync
```

### 3. 配置环境变量

```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑 .env 文件，配置数据库等信息
vim .env
```

**必要配置**:
```bash
# 应用配置
PYTHONDL_ENV=development
PYTHONDL_APP_DEBUG=true

# 服务器配置
PYTHONDL_SERVER_HOST=127.0.0.1
PYTHONDL_SERVER_PORT=8000

# 数据库配置
PYTHONDL_DATABASE_HOST=127.0.0.1
PYTHONDL_DATABASE_PORT=3306
PYTHONDL_DATABASE_USER=python
PYTHONDL_DATABASE_PASSWORD=123456
PYTHONDL_DATABASE_NAME=py_demo

# Redis 配置（可选）
PYTHONDL_REDIS_HOST=127.0.0.1
PYTHONDL_REDIS_PORT=6379
PYTHONDL_REDIS_PASSWORD=123456
```

### 4. 初始化数据库

```bash
# 创建数据库
mysql -u root -p -e "CREATE DATABASE py_demo CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 执行数据库迁移
alembic upgrade head
```

### 5. 启动后端服务

#### 开发环境

```bash
# 使用 Uvicorn 启动（热重载）
uvicorn app:app --reload --host 127.0.0.1 --port 8009
```

#### 生产环境

```bash
# 使用 Gunicorn 启动
gunicorn app:app -c gunicorn.conf.py
```

### 6. 启动前端服务

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 生产构建
npm run build
```

### 7. 访问系统

- **前端**: http://localhost:3000
- **后端 API**: http://localhost:8009
- **API 文档**: http://localhost:8009/docs
- **ReDoc**: http://localhost:8009/redoc

---

## 📁 项目结构

```
PythonDL/
├── app/                          # 后端应用
│   ├── admin/                    # 系统管理模块
│   ├── api/                      # API 接口层
│   ├── models/                   # 数据模型层
│   ├── schemas/                  # 数据验证层
│   ├── services/                 # 业务服务层
│   └── core/                     # 核心基础设施
├── frontend/                     # 前端应用
│   ├── src/
│   │   ├── api/                  # API 客户端
│   │   ├── components/           # UI 组件
│   │   ├── pages/                # 页面组件
│   │   └── layouts/              # 布局组件
│   └── dist/                     # 构建输出
├── alembic/                      # 数据库迁移
├── config/                       # 配置文件
├── docs/                         # 文档目录
├── logs/                         # 日志目录
├── runtimes/                     # 运行时目录
├── temps/                        # 临时文件
├── data/                         # 数据文件
├── files/                        # 导出文件
├── tests/                        # 测试目录
├── .env                          # 环境变量
├── pyproject.toml                # 项目配置
├── settings.toml                 # Dynaconf 配置
├── alembic.ini                   # Alembic 配置
├── gunicorn.conf.py              # Gunicorn 配置
└── ReadMe.md                     # 项目说明
```

详细架构文档：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 📚 文档

| 文档 | 描述 | 链接 |
|------|------|------|
| **架构文档** | 项目架构设计、技术栈说明 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| **API 文档** | Swagger 接口文档 | http://localhost:8009/docs |
| **测试报告** | API 测试报告 | [tests/API_TEST_REPORT.md](tests/API_TEST_REPORT.md) |
| **性能报告** | 性能测试报告 | [tests/performance/PERFORMANCE_REPORT.md](tests/performance/PERFORMANCE_REPORT.md) |
| **优化指南** | 性能优化指南 | [tests/performance/OPTIMIZATION_GUIDE.md](tests/performance/OPTIMIZATION_GUIDE.md) |

---

## 🧪 测试

### 运行单元测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/api/ -v

# 运行性能测试
python tests/performance/run_all_tests.py

# 生成覆盖率报告
pytest --cov=app --cov-report=html
```

### 测试覆盖率

- **总体覆盖率**: > 80%
- **核心模块**: > 90%
- **API 接口**: 100%

---

## 📊 API 接口

### RESTful 规范

```
GET    /api/v1/{resource}          # 获取列表
POST   /api/v1/{resource}          # 创建资源
GET    /api/v1/{resource}/{id}     # 获取详情
PUT    /api/v1/{resource}/{id}     # 更新资源
PATCH  /api/v1/{resource}/{id}     # 部分更新
DELETE /api/v1/{resource}/{id}     # 删除资源
```

### 统一响应格式

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

### 分页响应

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [],
    "total": 100,
    "page": 1,
    "page_size": 20,
    "total_pages": 5
  }
}
```

### 接口统计

| 模块 | 接口数 | 状态 |
|------|--------|------|
| 系统管理 | 40+ | ✅ |
| 金融分析 | 15+ | ✅ |
| 气象分析 | 12+ | ✅ |
| 看相算命 | 20+ | ✅ |
| 消费分析 | 18+ | ✅ |
| 爬虫采集 | 10+ | ✅ |
| **总计** | **115+** | ✅ |

---

## 🔐 安全机制

### 已实现的安全特性

1. **密码加密**: bcrypt 哈希，加盐存储
2. **JWT 认证**: 双令牌机制（Access + Refresh）
3. **SQL 注入防护**: ORM 参数化查询
4. **XSS 防护**: 输入验证和 HTML 转义
5. **CSRF 防护**: Token 验证
6. **速率限制**: API 请求限流（1000 次/分钟）
7. **CORS 控制**: 可配置的跨域策略
8. **权限控制**: RBAC 模型，细粒度权限

### 令牌机制

| 令牌类型 | 有效期 | 用途 |
|---------|--------|------|
| Access Token | 30 分钟 | API 访问 |
| Refresh Token | 7 天 | 刷新 Access Token |
| Session Token | 24 小时 | 会话保持 |

---

## 🚀 部署

### Docker 部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 生产环境配置

1. **配置环境变量**
2. **配置 Nginx 反向代理**
3. **配置 SSL 证书**
4. **配置日志轮转**
5. **配置监控告警**

详细部署文档：[docs/deployment.md](docs/deployment.md)

---

## 📈 性能指标

### 基准测试结果

| 指标 | 数值 | 等级 |
|------|------|------|
| API P95 响应时间 | < 200ms | 优秀 |
| API 成功率 | > 99.9% | 优秀 |
| 吞吐量 | > 500 req/s | 优秀 |
| 缓存命中率 | > 90% | 优秀 |
| 前端加载时间 | < 1s | 优秀 |

### 优化建议

详见：[tests/performance/OPTIMIZATION_GUIDE.md](tests/performance/OPTIMIZATION_GUIDE.md)

---

## 🤝 贡献指南

### 开发流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 规范
- 使用 Black 格式化代码
- 使用 Flake8 检查代码
- 编写单元测试

---

## 📄 License

本项目采用 MIT 协议开源，详见 [LICENSE](LICENSE) 文件。

---

## 📞 联系方式

- **项目地址**: https://github.com/yourusername/PythonDL
- **问题反馈**: https://github.com/yourusername/PythonDL/issues
- **邮箱**: your.email@example.com

---

## 🙏 致谢

感谢以下开源项目：

- [FastAPI](https://fastapi.tiangolo.com/)
- [Vue.js](https://vuejs.org/)
- [TailwindCSS](https://tailwindcss.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [TensorFlow](https://www.tensorflow.org/)

---

<div align="center">

**Made with ❤️ by PythonDL Team**

⭐ Star this repo if you like it!

</div>
