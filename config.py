"""
配置管理模块

集中管理数据库连接和 LLM API 配置，所有环境变量和常量在此统一定义。
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ==================== 数据库配置 ====================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3307)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "123456"),
    "database": os.getenv("DB_NAME", "chatbi_mvp"),
    "charset": "utf8mb4"
}


# ==================== LLM 配置 ====================
LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "model": os.getenv("LLM_MODEL", "gpt-4"),
    "temperature": 0.1,
    "max_tokens": 1000
}
