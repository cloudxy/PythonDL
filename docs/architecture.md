# PythonDL 项目架构文档

## 📋 目录

1. [项目概述](#项目概述)
2. [架构设计](#架构设计)
3. [技术栈](#技术栈)
4. [目录结构](#目录结构)
5. [核心模块](#核心模块)
6. [数据流](#数据流)
7. [部署架构](#部署架构)

---

## 项目概述

**PythonDL** 是一个全栈智能分析平台，集成了系统管理、金融分析、气象分析、看相算命、消费分析和爬虫采集六大核心模块。

### 核心特性

- 🎯 **模块化设计**: 清晰的功能模块划分
- 🚀 **高性能架构**: FastAPI + SQLAlchemy + MySQL
- 🎨 **顶级 UI**: Vue 3 + TailwindCSS
- 🤖 **AI/ML 集成**: TensorFlow, XGBoost, scikit-learn
- 🔐 **安全可靠**: JWT 认证，RBAC 权限
- 📊 **数据可视化**: 丰富的图表组件

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────┐
│          Presentation Layer             │
│         (Frontend - Vue 3 + Vite)       │
└─────────────────────────────────────────┘
                  ↓ HTTP/REST API
┌─────────────────────────────────────────┐
│           API Layer (FastAPI)           │
│      /api/v1/{module}/{resource}        │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         Service Layer (Business)        │
│      业务逻辑、数据处理、算法实现        │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│          Data Access Layer              │
│      (SQLAlchemy ORM + MySQL)           │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│           Database (MySQL)              │
└─────────────────────────────────────────┘
```

### 设计原则

1. **单一职责**: 每个模块专注于特定功能
2. **依赖倒置**: 高层模块不依赖低层模块具体实现
3. **接口隔离**: 细粒度的接口设计
4. **开闭原则**: 对扩展开放，对修改关闭

---

## 技术栈

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
| **缓存** | Redis 5.0+ | 内存缓存 |
| **日志** | logging + structlog | 日志记录 |
| **Web 服务器** | Gunicorn 21.2+ | WSGI 服务器 |
| **AI/ML** | TensorFlow 2.20+ | 深度学习 |
| **机器学习** | XGBoost 3.1+ | 机器学习 |
| **数据分析** | pandas 2.3+ | 数据处理 |

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

---

## 目录结构

```
PythonDL/
├── app/                          # 应用主目录
│   ├── admin/                    # 系统管理模块
│   │   ├── api/                  # 管理接口
│   │   └── services/             # 管理服务
│   ├── api/                      # API 接口层
│   │   └── v1/                   # v1 版本
│   ├── models/                   # 数据模型层
│   │   ├── admin/                # 管理模块模型
│   │   ├── finance/              # 金融模块模型
│   │   ├── weather/              # 气象模块模型
│   │   ├── fortune/              # 算命模块模型
│   │   └── consumption/          # 消费模块模型
│   ├── schemas/                  # 数据验证层
│   ├── services/                 # 业务服务层
│   └── core/                     # 核心基础设施
├── frontend/                     # 前端项目
│   ├── src/
│   │   ├── api/                  # API 客户端
│   │   ├── components/           # UI 组件
│   │   ├── pages/                # 页面组件
│   │   └── layouts/              # 布局组件
│   └── dist/                     # 构建输出
├── alembic/                      # 数据库迁移
├── config/                       # 配置文件
├── tests/                        # 测试目录
├── logs/                         # 日志目录
├── runtimes/                     # 运行时目录
└── data/                         # 数据文件
```

---

## 核心模块

### 1. 系统管理模块

**功能**: 用户管理、角色管理、权限管理、系统配置、日志管理、仪表盘

**API 路由**: `/api/v1/admin/*`

**关键文件**:
- `app/admin/api/users.py`
- `app/services/admin/user_service.py`
- `app/models/admin/user.py`

### 2. 金融分析模块

**功能**: 股票管理、股票预测、风险评估

**API 路由**: `/api/v1/finance/*`

**关键文件**:
- `app/api/v1/finance.py`
- `app/services/finance/stock_service.py`
- `app/models/finance/stock.py`

### 3. 气象分析模块

**功能**: 气象站点管理、气象数据管理、气象预测

**API 路由**: `/api/v1/weather/*`

**关键文件**:
- `app/api/v1/weather.py`
- `app/services/weather/weather_service.py`
- `app/models/weather/weather.py`

### 4. 看相算命模块

**功能**: 风水、面相、八字、周易、星座、运势管理

**API 路由**: `/api/v1/fortune/*`

**关键文件**:
- `app/api/v1/fortune.py`
- `app/services/fortune/bazi_service.py`
- `app/models/fortune/fortune_telling.py`

### 5. 消费分析模块

**功能**: GDP 数据、人口数据、经济指标、小区数据、消费预测

**API 路由**: `/api/v1/consumption/*`

**关键文件**:
- `app/api/v1/consumption.py`
- `app/services/consumption/gdp_service.py`
- `app/models/consumption/gdp_data.py`

### 6. 爬虫采集模块

**功能**: 股票采集、气象采集、算命采集、消费采集

**API 路由**: `/api/v1/crawler/*`

**关键文件**:
- `app/api/v1/crawler.py`
- `app/services/crawler/data_crawler.py`
- `app/models/admin/crawler_task.py`

---

## 数据流

### 请求处理流程

```
1. 客户端发送 HTTP 请求
   ↓
2. FastAPI 路由匹配
   ↓
3. 依赖注入（认证、权限）
   ↓
4. Schema 数据验证
   ↓
5. Service 业务逻辑处理
   ↓
6. Model 数据库操作
   ↓
7. 返回响应数据
```

### 数据查询流程

```
1. API 接收请求
   ↓
2. Service 层处理业务逻辑
   ↓
3. SQLAlchemy ORM 查询
   ↓
4. MySQL 执行查询
   ↓
5. 结果返回并序列化
   ↓
6. 响应给客户端
```

---

## 部署架构

### 开发环境

```
┌─────────────┐
│   Nginx     │
│  (Reverse   │
│   Proxy)    │
└──────┬──────┘
       │
       ├──→ http://localhost:3000 (Frontend)
       │
       └──→ http://localhost:8009 (Backend)
                  │
                  └──→ MySQL:3306
```

### 生产环境

```
┌─────────────┐
│   Nginx     │
│  (Load      │
│  Balancer)  │
└──────┬──────┘
       │
       ├──→ Frontend (Static Files)
       │
       └──→ Backend Cluster (Gunicorn)
                  │
                  ├──→ MySQL (Master-Slave)
                  │
                  └──→ Redis (Cache)
```

---

## 安全机制

### 认证授权

- JWT 双令牌机制（Access Token + Refresh Token）
- RBAC 权限模型
- 接口级权限控制

### 数据安全

- 密码 bcrypt 加密
- SQL 注入防护（ORM 参数化）
- XSS 防护
- CSRF 防护

### 速率限制

- API 请求限流
- IP 黑名单
- 用户级限流

---

## 性能优化

### 数据库优化

- 索引优化
- 查询优化
- 连接池管理
- 读写分离

### 缓存策略

- Redis 内存缓存
- 多级缓存
- 缓存预热
- 缓存失效策略

### 前端优化

- 代码分割
- 懒加载
- CDN 加速
- 图片优化

---

## 监控与日志

### 日志系统

- 按天流转
- 分类存储（app, error, gunicorn）
- 结构化日志
- 日志保留 30 天

### 性能监控

- API 响应时间
- 数据库查询性能
- 缓存命中率
- 错误率统计

---

*最后更新：2026-03-13*  
*版本：v2.0.0*
