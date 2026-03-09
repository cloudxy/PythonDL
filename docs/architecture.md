# PythonDL 项目架构文档

## 项目概述

PythonDL 是一个集成了多种功能模块的全栈智能分析平台，包括系统管理、金融分析、气象分析、看相算命分析、消费分析和爬虫采集等模块。项目采用现代化的技术栈，支持数据采集、分析、预测等功能。

## 技术栈

### 后端技术
- **Python 3.13+** - 主要开发语言
- **FastAPI** - 现代高性能Web框架
- **SQLAlchemy 2.0** - ORM框架
- **Alembic** - 数据库迁移工具
- **Dynaconf** - 配置管理
- **Gunicorn** - WSGI服务器
- **MySQL** - 主数据库
- **Redis** - 缓存数据库（可选）
- **Pydantic** - 数据验证

### 前端技术
- **Vue 3** - 前端框架
- **Vite** - 构建工具
- **Tailwind CSS** - 样式框架
- **Vue Router** - 路由管理
- **Pinia** - 状态管理

### AI/ML技术
- **TensorFlow** - 深度学习框架
- **XGBoost** - 梯度提升框架
- **Scikit-learn** - 机器学习库
- **NumPy** - 数值计算
- **Pandas** - 数据分析

## 目录结构

```
PythonDL/
├── alembic/                    # 数据库迁移
│   ├── versions/              # 迁移版本文件
│   └── env.py                 # Alembic配置
├── app/                       # 应用主目录
│   ├── admin/                 # 系统管理模块
│   ├── api/                   # API接口
│   │   └── v1/               # v1版本API
│   │       ├── admin.py      # 系统管理API
│   │       ├── auth.py       # 认证API
│   │       ├── consumption.py # 消费分析API
│   │       ├── crawler.py    # 爬虫采集API
│   │       ├── finance.py    # 金融分析API
│   │       ├── fortune.py    # 看相算命API
│   │       └── weather.py    # 气象分析API
│   ├── core/                  # 核心功能
│   │   ├── cache.py          # 缓存管理
│   │   ├── config.py         # 配置管理
│   │   ├── database.py       # 数据库连接
│   │   ├── exceptions.py     # 异常处理
│   │   ├── logger.py         # 日志系统
│   │   ├── monitoring.py     # 性能监控
│   │   ├── rate_limit.py     # 速率限制
│   │   └── security.py       # 安全模块
│   ├── models/                # 数据模型
│   │   ├── admin/            # 系统管理模型
│   │   ├── consumption/      # 消费分析模型
│   │   ├── finance/          # 金融分析模型
│   │   ├── fortune/          # 看相算命模型
│   │   └── weather/          # 气象分析模型
│   ├── schemas/               # 数据验证
│   ├── services/              # 业务逻辑
│   │   ├── admin/            # 系统管理服务
│   │   ├── consumption/      # 消费分析服务
│   │   ├── crawler/          # 爬虫采集服务
│   │   ├── finance/          # 金融分析服务
│   │   ├── fortune/          # 看相算命服务
│   │   └── weather/          # 气象分析服务
│   ├── static/                # 静态资源
│   └── __init__.py           # 应用入口
├── data/                      # 数据文件
├── docs/                      # 项目文档
├── files/                     # 导出文件
│   ├── exports/              # 导出文件
│   ├── images/               # 图片文件
│   └── videos/               # 视频文件
├── frontend/                  # 前端代码
│   └── src/
│       ├── components/       # 组件
│       ├── layouts/          # 布局
│       ├── pages/            # 页面
│       ├── router/           # 路由
│       ├── stores/           # 状态管理
│       └── main.js           # 入口文件
├── logs/                      # 日志文件
│   ├── app/                  # 应用日志
│   ├── error/                # 错误日志
│   └── gunicorn/             # Gunicorn日志
├── runtimes/                  # 运行时文件
│   ├── cache/                # 缓存文件
│   └── sessions/             # 会话文件
├── temps/                     # 临时文件
├── tests/                     # 测试代码
├── utils/                     # 工具函数
├── .env.example              # 环境变量示例
├── .gitignore                # Git忽略文件
├── alembic.ini               # Alembic配置
├── app.py                    # 应用入口
├── docker-compose.yml        # Docker编排
├── Dockerfile                # Docker镜像
├── gunicorn.conf.py          # Gunicorn配置
├── pyproject.toml            # 项目配置
├── ReadMe.md                 # 项目说明
├── settings.toml             # 应用配置
└── uv.lock                   # 依赖锁定
```

## 功能模块

### 1. 系统管理模块

#### 用户管理
- 用户列表、创建、编辑、删除
- 用户角色分配
- 用户状态管理

#### 角色管理
- 角色列表、创建、编辑、删除
- 角色权限分配

#### 权限管理
- 权限列表、创建、编辑、删除
- 权限树形结构

#### 系统配置
- 配置列表、创建、编辑、删除
- 系统名称、样式配置

#### 日志管理
- 操作日志列表
- 日志查询和过滤

#### 仪表盘
- 系统整体情况展示
- 关键指标统计

### 2. 金融分析模块

#### 股票管理
- 股票基础信息管理
- 股票行情数据管理
- 数据采集和更新

#### 股票预测
- 基于历史数据的股价预测
- LSTM、XGBoost等多种模型
- 预测结果可视化

#### 股票风险评估
- 波动率计算
- VaR风险价值
- 夏普比率
- 最大回撤
- 风险建议

### 3. 气象分析模块

#### 气象管理
- 气象站点管理
- 气象数据管理
- 数据采集和更新

#### 气象预测
- 基于历史数据的天气预测
- 温度、湿度、降水预测
- 预测结果可视化

### 4. 看相算命分析模块

#### 数据管理
- 风水数据管理
- 面相数据管理
- 八字数据管理
- 周易数据管理
- 星座数据管理
- 运势数据管理

#### 分析功能
- 综合分析
- 专业解读
- 个性化建议

### 5. 消费分析模块

#### 数据管理
- GDP数据管理
- 人口数据管理
- 经济指标管理
- 小区数据管理

#### 消费预测
- 宏观消费趋势预测
- 经济指标预测
- 政策建议

### 6. 爬虫采集模块

#### 数据采集
- 股票数据采集
- 气象数据采集
- 看相算命数据采集
- 宏观消费数据采集

## API设计

### RESTful API规范

所有API遵循RESTful设计规范：

- `GET` - 获取资源
- `POST` - 创建资源
- `PUT` - 更新资源
- `DELETE` - 删除资源

### API版本控制

API使用版本控制，当前版本为v1：
- `/api/v1/auth/*` - 认证相关
- `/api/v1/admin/*` - 系统管理
- `/api/v1/finance/*` - 金融分析
- `/api/v1/weather/*` - 气象分析
- `/api/v1/fortune/*` - 看相算命
- `/api/v1/consumption/*` - 消费分析
- `/api/v1/crawler/*` - 爬虫采集

### 统一响应格式

```json
{
  "success": true,
  "data": {},
  "message": "操作成功"
}
```

### 错误响应格式

```json
{
  "success": false,
  "error_code": "ERROR_CODE",
  "message": "错误信息"
}
```

## 数据库设计

### 核心表结构

#### 用户相关
- `users` - 用户表
- `roles` - 角色表
- `permissions` - 权限表
- `role_permissions` - 角色权限关联表

#### 系统相关
- `system_configs` - 系统配置表
- `operation_logs` - 操作日志表

#### 金融相关
- `stock_basics` - 股票基础信息表
- `stocks` - 股票行情数据表
- `stock_predictions` - 股票预测表
- `stock_risk_assessments` - 股票风险评估表

#### 气象相关
- `weather_stations` - 气象站点表
- `weather_data` - 气象数据表
- `weather_forecasts` - 气象预测表

#### 看相算命相关
- `feng_shui` - 风水数据表
- `face_readings` - 面相数据表
- `bazi` - 八字数据表
- `zhou_yi` - 周易数据表
- `constellations` - 星座数据表
- `fortune_tellings` - 运势数据表

#### 消费相关
- `gdp_data` - GDP数据表
- `population_data` - 人口数据表
- `economic_indicators` - 经济指标表
- `community_data` - 小区数据表
- `consumption_forecasts` - 消费预测表

## 安全设计

### 认证授权
- JWT令牌认证
- 刷新令牌机制
- 基于角色的权限控制

### 安全措施
- 密码加密存储
- 速率限制
- 输入验证
- CORS配置
- SQL注入防护
- XSS防护

## 性能优化

### 缓存策略
- 文件缓存
- Redis缓存（可选）
- 缓存预热
- 缓存过期管理

### 数据库优化
- 索引优化
- 连接池
- 查询优化
- 分页查询

### 异步处理
- 后台任务
- 异步API
- 定时任务

## 部署方案

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

### 环境搭建

```bash
# 安装UV
pip install uv

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env

# 运行数据库迁移
alembic upgrade head

# 启动开发服务器
uv run uvicorn app:app --reload
```

### 代码规范

- 使用Black进行代码格式化
- 使用Flake8进行代码检查
- 使用MyPy进行类型检查
- 遵循PEP 8规范

### 测试

```bash
# 运行测试
uv run pytest

# 运行测试并生成覆盖率
uv run pytest --cov=app --cov-report=html
```

## 监控与日志

### 日志系统
- 按天流转
- 区分普通日志和错误日志
- 自动压缩和清理

### 性能监控
- 请求响应时间
- 慢请求告警
- 资源使用监控

## 未来规划

1. 增加更多AI模型集成
2. 扩展数据采集范围
3. 优化用户体验
4. 增加更多分析功能
5. 支持多语言
6. 移动端适配
