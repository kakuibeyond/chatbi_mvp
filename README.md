## 模块说明

### 核心系统模块（第 6-12 课）

| 模块 | 用途 | 关键接口 |
|------|------|---------|
| `config.py` | 集中管理数据库和 LLM API 配置 | `DB_CONFIG`、`LLM_CONFIG` |
| `query_parser.py` | 用户输入校验 | `parse_query(question) → str` |
| `prompt_builder.py` | Prompt 组装（Schema + Rules + Few-shot + 指标知识） | `build_prompt(question, ...) → (system_msg, prompt)` |
| `llm_client.py` | LLM API 调用（同步文本/SQL、流式、Embedding） | `generate_text()`、`generate_sql()`、`generate_sql_stream()`、`get_embedding()` |
| `database.py` | MySQL 连接与 SQL 执行 | `execute_sql(sql) → results` |
| `result_formatter.py` | 结果格式化为表格/文本 | `format_result(results) → str` |
| `main.py` | 系统主入口（CLI + ChatBISystem 类） | `ChatBISystem.run(question)`、`.run_stream(question)`、内部统一走 `_resolve_indicator_context()` |
| `api_service.py` | FastAPI 服务（同步 + SSE 流式接口） | `POST /api/v1/query`、`POST /api/v1/query/stream`，支持 `use_schema_linking` / `use_indicator_rag` 开关 |
| `schema_generator.py` | 从数据库 information_schema 自动提取表结构 | `generate_schema() → str` |

### Agent 预备模块（第 22 课）

| 模块 | 用途 | 关键接口 |
|------|------|---------|
| `query_decomposer.py` | 将复杂分析问题拆解为结构化子任务列表 | `build_decomposition_prompt(question) → (system_msg, prompt)`、`QueryDecomposer.decompose(question) → dict` |

### Agent 执行主链（第 23-25 课）

| 模块 | 用途 | 关键接口 |
|------|------|---------|
| `agent_planner.py` | 实现 Plan Generator、Step Executor、Result Summarizer，并串成可运行的 Plan-and-Execute Agent；第 24 课补充中间结果引用、重试、失败跳过、执行状态跟踪，第 25 课新增 `report` 输出并接入 `ReportGenerator` | `PlanGenerator.build_plan()`、`StepExecutor.execute_plan()`、`PlanAndExecuteAgent.run()`、`uv run agent_planner.py --plan-only`、`uv run agent_planner.py --max-retries 1 --failure-policy skip --storage-backend temp_table` |
| `report_generator.py` | 把多步执行结果整理为结构化分析报告；优先走 LLM 结构化 JSON，再渲染 Markdown，格式异常时回退模板化报告 | `ReportGenerator.generate()`、`AnalysisReport` |
| `tests/test_agent_planner.py` | 验证计划生成、依赖上下文拼接、执行摘要输出，以及第 24-25 课新增的重试 / 跳过 / 状态记录 / temp table 存储 / report 输出 | `uv run pytest tests/test_agent_planner.py` |
| `tests/test_report_generator.py` | 验证报告解析与模板回退逻辑 | `uv run pytest tests/test_report_generator.py` |

### 错误分析与评估模块（第 7-8 课）

| 模块 | 用途 | 运行方式 |
|------|------|---------|
| `error_analyzer.py` | SQL 错误分类（启发式规则 + LLM 分析） | `uv run error_analyzer.py` |
| `evaluator.py` | Execution Accuracy + Exact Match 自动化评估 | `uv run evaluator.py` |
| `test_cases.json` | 三层难度测试用例集 | 被 evaluator.py 读取 |

### 指标知识模块（第 9 课 + 第 19 课升级）

| 模块 | 用途 | 关键接口 |
|------|------|---------|
| `indicator_knowledge.py` | 指标识别（关键词匹配，第 9 课原版） | `get_indicator_context(question) → {"detected_indicators", "indicator_block"}` |
| `indicators.json` | 5 个核心指标定义（第 9 课原版） | JSON 格式 |
| `indicator_retriever.py` | 指标 RAG 检索（语义检索，第 19 课升级版） | `retrieve_indicator_context(query) → {"detected_indicators", "indicator_block"}`、`retrieve_indicators(query) → [dict]` |
| `indicators_full.json` | 13 个核心指标完整知识库（含分层、依赖链、SQL 模板） | JSON 格式 |

### 向量检索与 Schema Linking 模块（第 14-20 课，第四阶段）

| 模块 | 用途 | 关键接口 |
|------|------|---------|
| `vector_search_demo.py` | 手写向量检索演示 | `search_tables(query, index, client) → [(table, score)]` |
| `table_retriever.py` | 表级召回（LangChain + ChromaDB 工程化版本） | `retrieve_tables(query, top_k, threshold) → [dict]`、`build_index()` |
| `field_matcher.py` | 字段语义匹配（向量相似度 + 业务规则混合） | `match_fields(query, tables) → [dict]`、`retrieve_schema(query) → dict` |
| `join_resolver.py` | 多表 Join 路径自动推理（锚表选择 + BFS 图算法） | `select_anchor(query, candidate_tables) → (anchor, reason)`、`resolve_joins(anchor, target_tables) → dict` |
| `schema_linker.py` | Schema Linking Pipeline 编排（串联表/字段/Join） | `schema_link(query) → dict`、`build_dynamic_prompt_schema(query) → str` |
| `chroma_db/` | ChromaDB 持久化向量数据目录 | 自动管理，勿手动编辑 |

### 历史脚本

| 模块 | 用途 |
|------|------|
| `text2sql_v0.py` | 第 5 课：Zero-shot 单文件原型 |
| `text2sql_v1.py` | 第 5 课：Few-shot 增强版本 |
| `text2sql_v2.py` | 第 5 课：COT + 结构化约束版本 |

---

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

### Git 提交规范

- **每节课独立提交**：每节课的 code 代码在撰写完成、运行成功、测试没有问题之后，需要单独做一条 commit。commit message 格式为 `lesson-<课程序号>: <简述本课代码变更>`，例如 `lesson-14: 手写向量检索 demo（vector_search_demo.py）`
- **可随时切换版本**：通过 `git log` 查看历史，可以随时 checkout 到任意一课完成时的代码状态，用于课前环境准备或复现演示
- **不跨课混合提交**：即使两课的代码有依赖关系，也必须分开提交。前置课的代码先提交，后续课基于前置课的 commit 继续开发
