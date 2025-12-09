### 安装UV
```shell
# Linux/Mac（一键安装）
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Windows（PowerShell）
PowerShell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

# 验证安装（输出版本即成功）
uv --version
```


### 初始化
```shell
# 用uv创建虚拟环境
uv init my_project --python 3.13
```


### 安装依赖
```shell
# 安装生产依赖（如 requests），自动更新 pyproject.toml 和 poetry.lock
uv add requests

# 读取 poetry.lock 安装所有依赖
uv install
```

### 升级依赖
```shell
# 升级指定依赖，更新 poetry.lock
uv upgrade requests
# 升级所有依赖（谨慎使用，可能引入版本冲突）
uv upgrade
```


### 打包与发布
```shell
# 打包项目（生成 wheel 包和源码包，存放在 dist/ 目录）
poetry build

# 发布到 PyPI（需先配置 PyPI 密钥，参考之前的 Poetry 敏感配置）
poetry publish
```

### 运行项目
```shell
# 方式1：用 uv 直接运行（无需激活环境，速度快）
uv run python main.py

# 方式2：用 Poetry 运行（自动关联 uv 环境）
poetry run python main.py
```



