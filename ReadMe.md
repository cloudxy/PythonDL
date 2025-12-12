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


### 配置github验证
```text
步骤 1：生成 GitHub Personal Access Token（PAT）
打开 GitHub 官网，登录你的账号（cloudxy）；
点击右上角头像 →「Settings」（设置）→ 左侧栏拉到最下方，点击「Developer settings」（开发者设置）；
左侧栏点击「Personal access tokens」→「Tokens (classic)」→ 右上角「Generate new token」（生成新令牌）；
填写令牌信息：
「Note」：备注令牌用途（如「Git 推送认证」，方便后续管理）；
「Expiration」：设置令牌有效期（建议选「30 days」或「Custom」，定期更换更安全）；
「Scopes」（权限）：必须勾选「repo」相关所有权限（否则无法推送代码），其他权限按需勾选（无需额外勾选）；
拉到页面最下方，点击「Generate token」（生成令牌）；
关键：令牌生成后会仅显示一次，立即复制保存（比如记在备忘录里），关闭页面后就再也看不到了！

步骤 2：用 PAT 重新执行 git push
回到终端，重新执行 git push，会再次提示输入用户名和密码：
「Username for 'https://github.com'」：输入你的 GitHub 用户名
「Password for 'https://cloudxy@github.com'」：粘贴步骤 1 生成的 PAT 令牌（不是你的 GitHub 密码！），粘贴后按回车即可
```


### git简单使用
```text
# 初始化git（若未初始化）
# 初始化 Git 仓库（仅首次执行）
git init
# 直接添加新的 origin 地址
git remote add origin https://xxx

# 查看当前的 remote.origin.url
git remote -v

# 新的远程仓库地址
git remote set-url origin https://xxx

# 查看本地和远程所有分支
git branch -a

# 拉去远程所有分支的最新状态
git fetch origin

# 从本地 master 分支创建 main 分支（继承 master 所有代码）
# 等价于：git branch main + git checkout main
git checkout -b main  

# 将本地 main 分支推送到远程（创建远程 main 分支）
# -u 表示关联本地 main 和远程 main（后续可直接 git push）
git push -u origin main 

# 切换到本地 main 分支（若本地没有，先创建：git checkout -b main）
git checkout main

# 拉取远程 main 分支的最新代码（避免合并时冲突）
git pull origin main
# 把 master 的代码合并到 main
git merge master
# 合并 master 到 main 并允许无关历史
git merge master --allow-unrelated-histories
```

### 安装包速度慢
```shell
# 在pyproject.toml文件中，添加一下内容
# 主索引源（优先级最高）
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
# 备用索引源（主源找不到时用）
extra-index-url = ["https://pypi.org/simple/"]
# 可选：禁用默认的 PyPI 源（避免冗余）
no-default-index = true
```

### tensorflow下载较慢
```shell
# 核心命令：清华源（对 TensorFlow 镜像支持最好）+ 并行下载 + 详细日志（验证是否走镜像）
uv add tensorflow --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --verbose
```

### 在已创建的venv中手动安装pip
```shell
# 1. 激活venv（可选，但更直观）
source ./.venv/bin/activate

# 2. 用Python内置模块安装并升级pip
python -m ensurepip --upgrade

# 3. 验证pip是否存在
# 激活venv后直接用，或：./.venv/bin/pip --version
pip --version
```