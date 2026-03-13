# PythonDL API 接口完整清单

## 基础信息

- **Base URL**: `/api/v1`
- **认证方式**: JWT Bearer Token
- **文档地址**: `/docs` (Swagger UI)
- **备用文档**: `/redoc` (ReDoc)

---

## 认证模块 (/api/v1/auth)

### 用户认证
| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| POST | `/login` | 用户登录 | ❌ |
| POST | `/register` | 用户注册 | ❌ |
| POST | `/refresh` | 刷新令牌 | ❌ |
| POST | `/logout` | 用户登出 | ✅ |
| POST | `/forgot-password` | 忘记密码 | ❌ |
| POST | `/reset-password` | 重置密码 | ❌ |
| GET | `/me` | 获取当前用户信息 | ✅ |

---

## 系统管理模块 (/api/v1/admin)

### 用户管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/users` | 获取用户列表 | ✅ |
| POST | `/users` | 创建用户 | `user:create` |
| GET | `/users/{user_id}` | 获取用户详情 | ✅ |
| PUT | `/users/{user_id}` | 更新用户 | `user:update` |
| DELETE | `/users/{user_id}` | 删除用户 | `user:delete` |

### 角色管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/roles` | 获取角色列表 | ✅ |
| POST | `/roles` | 创建角色 | `role:create` |
| GET | `/roles/{role_id}` | 获取角色详情 | ✅ |
| PUT | `/roles/{role_id}` | 更新角色 | `role:update` |
| DELETE | `/roles/{role_id}` | 删除角色 | `role:delete` |
| POST | `/roles/{role_id}/users/{user_id}` | 分配角色给用户 | ✅ |
| DELETE | `/roles/{role_id}/users/{user_id}` | 从用户移除角色 | ✅ |
| GET | `/roles/{role_id}/permissions` | 获取角色权限 | ✅ |

### 权限管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/permissions` | 获取权限列表 | ✅ |
| POST | `/permissions` | 创建权限 | `permission:create` |
| GET | `/permissions/{permission_id}` | 获取权限详情 | ✅ |
| PUT | `/permissions/{permission_id}` | 更新权限 | `permission:update` |
| DELETE | `/permissions/{permission_id}` | 删除权限 | `permission:delete` |
| POST | `/roles/{role_id}/permissions/{permission_id}` | 分配权限给角色 | ✅ |
| DELETE | `/roles/{role_id}/permissions/{permission_id}` | 从角色移除权限 | ✅ |
| GET | `/permissions/users/{user_id}` | 获取用户权限 | ✅ |

### 系统配置管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/configs` | 获取系统配置列表 | ✅ |
| POST | `/configs` | 创建系统配置 | `config:create` |
| GET | `/configs/{config_id}` | 获取系统配置详情 | ✅ |
| PUT | `/configs/{config_id}` | 更新系统配置 | `config:update` |
| DELETE | `/configs/{config_id}` | 删除系统配置 | `config:delete` |
| GET | `/configs/key/{config_key}` | 根据键获取配置值 | ✅ |

### 日志管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/logs` | 获取系统日志列表 | `log:view` |
| GET | `/logs/{log_id}` | 获取日志详情 | `log:view` |
| DELETE | `/logs/{log_id}` | 删除日志 | `log:delete` |
| DELETE | `/logs/cleanup/{days}` | 清理旧日志 | `log:delete` |
| GET | `/logs/errors/recent` | 获取错误日志 | `log:view` |
| GET | `/logs/user/{user_id}` | 获取用户操作日志 | `log:view` |
| GET | `/logs/module/{module}` | 获取模块日志 | `log:view` |

### 仪表盘
| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| GET | `/dashboard/overview` | 获取仪表盘概览 | ✅ |
| GET | `/dashboard/users` | 获取用户统计 | ✅ |
| GET | `/dashboard/logs` | 获取日志统计 | ✅ |
| GET | `/dashboard/health` | 获取系统健康状态 | ✅ |
| GET | `/dashboard/modules` | 获取模块统计 | ✅ |
| GET | `/dashboard/activities` | 获取最近活动 | ✅ |

---

## 金融分析模块 (/api/v1/finance)

### 股票基础信息管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/stocks/basic` | 获取股票基础信息列表 | ✅ |
| POST | `/stocks/basic` | 创建股票基础信息 | `stock:create` |
| GET | `/stocks/basic/{stock_id}` | 获取股票基础信息详情 | ✅ |
| PUT | `/stocks/basic/{stock_id}` | 更新股票基础信息 | `stock:update` |
| DELETE | `/stocks/basic/{stock_id}` | 删除股票基础信息 | `stock:delete` |

### 股票行情数据管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/stocks/data` | 获取股票行情数据列表 | ✅ |
| POST | `/stocks/data` | 创建股票行情数据 | `stock:create` |

### 股票预测与评估
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/stocks/predict` | 股票预测 | `stock:predict` |
| GET | `/stocks/risk/{ts_code}` | 股票风险评估 | ✅ |

---

## 气象分析模块 (/api/v1/weather)

### 气象站点管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/stations` | 获取气象站点列表 | ✅ |
| POST | `/stations` | 创建气象站点 | `weather:create` |
| GET | `/stations/{station_id}` | 获取气象站点详情 | ✅ |
| PUT | `/stations/{station_id}` | 更新气象站点 | `weather:update` |
| DELETE | `/stations/{station_id}` | 删除气象站点 | `weather:delete` |

### 气象数据管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/data` | 获取气象数据列表 | ✅ |
| POST | `/data` | 创建气象数据 | `weather:create` |

### 气象预测
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/forecast` | 气象预测 | `weather:forecast` |

---

## 看相算命模块 (/api/v1/fortune)

### 风水管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/feng-shui` | 获取风水数据列表 | ✅ |
| POST | `/feng-shui` | 创建风水数据 | `fortune:create` |
| PUT | `/feng-shui/{item_id}` | 更新风水数据 | `fortune:update` |
| DELETE | `/feng-shui/{item_id}` | 删除风水数据 | `fortune:delete` |

### 面相管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/face-reading` | 获取面相数据列表 | ✅ |
| POST | `/face-reading` | 创建面相数据 | `fortune:create` |
| PUT | `/face-reading/{item_id}` | 更新面相数据 | `fortune:update` |
| DELETE | `/face-reading/{item_id}` | 删除面相数据 | `fortune:delete` |

### 八字管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/bazi` | 获取八字数据列表 | ✅ |
| POST | `/bazi` | 创建八字数据 | `fortune:create` |

### 周易管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/zhou-yi` | 获取周易数据列表 | ✅ |

### 星座管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/constellation` | 获取星座数据列表 | ✅ |

### 运势管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/fortune-telling` | 获取运势数据列表 | ✅ |
| POST | `/fortune-telling` | 创建运势数据 | `fortune:create` |

### 综合分析
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/analyze` | 看相算命综合分析 | ✅ |

---

## 消费分析模块 (/api/v1/consumption)

### GDP 数据管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/gdp` | 获取 GDP 数据列表 | ✅ |
| POST | `/gdp` | 创建 GDP 数据 | `consumption:create` |
| PUT | `/gdp/{item_id}` | 更新 GDP 数据 | `consumption:update` |
| DELETE | `/gdp/{item_id}` | 删除 GDP 数据 | `consumption:delete` |

### 人口数据管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/population` | 获取人口数据列表 | ✅ |
| POST | `/population` | 创建人口数据 | `consumption:create` |
| PUT | `/population/{item_id}` | 更新人口数据 | `consumption:update` |
| DELETE | `/population/{item_id}` | 删除人口数据 | `consumption:delete` |

### 经济指标管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/economic-indicators` | 获取经济指标数据列表 | ✅ |
| POST | `/economic-indicators` | 创建经济指标数据 | `consumption:create` |

### 小区数据管理
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| GET | `/community` | 获取小区数据列表 | ✅ |
| POST | `/community` | 创建小区数据 | `consumption:create` |
| PUT | `/community/{item_id}` | 更新小区数据 | `consumption:update` |
| DELETE | `/community/{item_id}` | 删除小区数据 | `consumption:delete` |

### 消费预测
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/forecast` | 宏观消费预测 | `consumption:forecast` |

---

## 爬虫采集模块 (/api/v1/crawler)

### 股票数据采集
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/stock/start` | 启动股票数据采集 | `crawler:start` |
| GET | `/stock/status` | 获取股票数据采集状态 | ✅ |

### 气象数据采集
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/weather/start` | 启动气象数据采集 | `crawler:start` |
| GET | `/weather/status` | 获取气象数据采集状态 | ✅ |

### 看相算命数据采集
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/fortune/start` | 启动看相算命数据采集 | `crawler:start` |
| GET | `/fortune/status` | 获取看相算命数据采集状态 | ✅ |

### 宏观消费数据采集
| 方法 | 路径 | 描述 | 权限 |
|------|------|------|------|
| POST | `/consumption/start` | 启动宏观消费数据采集 | `crawler:start` |
| GET | `/consumption/status` | 获取宏观消费数据采集状态 | ✅ |

---

## 其他端点

### 健康检查
| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| GET | `/health` | 健康检查 | ❌ |
| GET | `/` | API 根路径 | ❌ |

---

## 认证说明

### 获取 Token
```bash
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

grant_type=password&username=your_username&password=your_password
```

### 使用 Token
```bash
GET /api/v1/admin/users
Authorization: Bearer YOUR_ACCESS_TOKEN
```

### Token 刷新
```bash
POST /api/v1/auth/refresh
Content-Type: application/x-www-form-urlencoded

refresh_token=YOUR_REFRESH_TOKEN
```

---

## 错误响应格式

```json
{
  "detail": "错误信息描述"
}
```

## 成功响应格式

### 单个资源
```json
{
  "id": 1,
  "field1": "value1",
  "field2": "value2"
}
```

### 列表资源
```json
[
  {
    "id": 1,
    "field1": "value1"
  },
  {
    "id": 2,
    "field1": "value2"
  }
]
```

### 带分页的列表
```json
{
  "success": true,
  "data": {
    "items": [...],
    "total": 100,
    "skip": 0,
    "limit": 20
  },
  "message": "获取成功"
}
```

---

## 通用查询参数

### 分页参数
- `skip`: 跳过记录数，默认 0
- `limit`: 返回记录数，默认 20，最大 100

### 过滤参数
- 各模块根据字段提供不同的过滤参数
- 支持模糊查询的字段使用 `%` 通配符

### 日期范围参数
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)

---

## 权限说明

### 权限格式
`resource:action`

### 常见权限
- `user:create` - 创建用户
- `user:update` - 更新用户
- `user:delete` - 删除用户
- `role:create` - 创建角色
- `role:update` - 更新角色
- `role:delete` - 删除角色
- `permission:create` - 创建权限
- `permission:update` - 更新权限
- `permission:delete` - 删除权限
- `config:create` - 创建配置
- `config:update` - 更新配置
- `config:delete` - 删除配置
- `log:view` - 查看日志
- `log:delete` - 删除日志
- `stock:create` - 创建股票
- `stock:update` - 更新股票
- `stock:delete` - 删除股票
- `stock:predict` - 股票预测
- `weather:create` - 创建气象数据
- `weather:update` - 更新气象数据
- `weather:delete` - 删除气象数据
- `weather:forecast` - 气象预测
- `fortune:create` - 创建算命数据
- `fortune:update` - 更新算命数据
- `fortune:delete` - 删除算命数据
- `consumption:create` - 创建消费数据
- `consumption:update` - 更新消费数据
- `consumption:delete` - 删除消费数据
- `consumption:forecast` - 消费预测
- `crawler:start` - 启动爬虫

---

## 使用示例

### 1. 用户登录
```bash
curl -X POST "http://localhost:8009/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password&username=admin&password=admin123"
```

### 2. 获取用户列表
```bash
curl -X GET "http://localhost:8009/api/v1/admin/users?skip=0&limit=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 3. 创建股票基础信息
```bash
curl -X POST "http://localhost:8009/api/v1/finance/stocks/basic" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_code": "000001.SZ",
    "symbol": "000001",
    "name": "平安银行",
    "industry": "银行"
  }'
```

### 4. 启动股票数据采集
```bash
curl -X POST "http://localhost:8009/api/v1/crawler/stock/start?days=30" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

**总计接口数量**: 100+
**文档版本**: 1.0.0
**最后更新**: 2026-03-13
