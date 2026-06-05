## 开发环境说明
- 本项目用 uv 作为 Python 环境管理工具

### 环境初始化步骤

代码仓库首次配置仅需执行一次：

```bash
# 1. 安装 uv（macOS / Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # 或 ~/.zshrc

# 验证安装
uv --version

# 2. 创建项目目录并初始化
mkdir chatbi-mvp
cd chatbi-mvp
uv init
uv venv
```

初始化完成后，项目目录结构如下：

```
chatbi-mvp/
├── .venv/              # 虚拟环境目录
├── .python-version     # Python 版本锁定
├── pyproject.toml      # 项目配置和依赖声明
├── uv.lock             # 依赖锁文件（自动管理，勿手动编辑）
└── README.md
```

配置国内镜像源

在 `pyproject.toml` 末尾添加：

```toml
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
```

### 安装依赖与运行脚本

```bash
# 安装本课所需依赖（示例）
uv add pymysql openai

# 查看依赖树
uv tree

# 在虚拟环境中运行 Python 脚本
uv run <脚本名>.py
```

### 高频 uv 命令速查

| 命令 | 作用 |
|---|---|
| `uv init` | 初始化新项目，创建 `pyproject.toml` |
| `uv venv` | 创建虚拟环境 |
| `uv add <包名>` | 安装依赖并写入 `pyproject.toml` |
| `uv remove <包名>` | 删除依赖 |
| `uv run <脚本.py>` | 在虚拟环境中运行脚本 |
| `uv lock` | 生成或更新 `uv.lock` 锁文件 |
| `uv sync` | 根据 `uv.lock` 同步依赖（团队协同时常用） |

### 代码书写规范

- **代码块必须标注语言**：所有 SQL、Python、Bash 代码块使用 `` ```sql `` / `` ```python `` / `` ```bash `` 标注
- **脚本需可直接运行**：示例代码应确保复制到 `chatbi-mvp` 项目目录后，执行 `uv run <脚本>.py` 即可运行，不应依赖未声明的外部环境
- **环境变量通过 .env 文件管理**：所有脚本统一使用 `python-dotenv` 自动加载项目根目录下的 `.env` 文件，不要在代码中硬写密钥，也不要依赖手动 `export` 环境变量。`.env` 文件不提交到 Git，通过 `.env.example` 提供模板
- **避免全局 Python**：统一使用 `uv run` 运行脚本，不推荐直接调用系统全局 Python 或 `python <脚本>.py`
