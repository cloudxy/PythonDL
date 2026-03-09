from dynaconf import Dynaconf
import os

# 获取当前环境（默认本地开发环境）
current_env = os.getenv("APP_ENV", "local")

# 配置目录
config_dir = "config"
default_config_dir = os.path.join(config_dir, "default")

# 配置文件路径
default_settings = os.path.join(default_config_dir, "settings.yml")


# 初始化Dynaconf（加载默认+当前环境+敏感配置）
settings = Dynaconf(
    envvar_prefix="PythonDL",  # 环境变量前缀（避免冲突）
    settings_files=[
        default_settings  # 先加载默认配置
    ],
    env_file=env_dotenv,    # 加载当前环境的敏感.env
    load_dotenv=True,       # 启用.env加载
    merge_enabled=True      # 启用配置合并（默认+环境配置叠加）
)

