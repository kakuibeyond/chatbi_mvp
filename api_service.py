"""
FastAPI 服务入口

将命令行版 ChatBI 系统封装为 REST API，供前端或其他服务调用。

第11课增强：
- 新增 POST /api/v1/query/stream SSE 流式端点
- 新增 CORS 中间件（前后端分离必须）
- 挂载 static 目录为静态文件服务（前端 HTML）
- 保留原有 /api/v1/query 同步端点不动，确保向后兼容
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import ChatBISystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("chatbi.api")

app = FastAPI(
    title="ChatBI MVP API",
    version="0.2.0",
    description="""
## ChatBI MVP API

企业级 ChatBI 最小可行产品的服务化接口。

### 接口概览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/query` | POST | 同步查询，一次性返回完整结果 |
| `/api/v1/query/stream` | POST | SSE 流式查询，逐步推送 SQL 和结果 |
| `/health` | GET | 健康检查 |

### SSE 流式接口事件类型

`/api/v1/query/stream` 返回的事件类型：

| 事件类型 | 说明 | data 结构 |
|----------|------|-----------|
| `sql_chunk` | SQL 文本片段，前端逐字拼接展示 | `{"content": "SELECT..."}` |
| `sql_done` | SQL 完整输出，可复制或二次处理 | `{"sql": "SELECT COUNT(*)..."}` |
| `result` | 查询结果 | `{"columns": [...], "rows": [...], "row_count": N}` |
| `error` | 异常信息 | `{"error": "...", "error_type": "..."}` |

### Apifox 导入方式

1. 启动服务后访问 `http://localhost:8000/openapi.json`
2. 在 Apifox 中选择「导入」→「URL 导入」→ 粘贴上面的地址
3. 或直接复制 JSON 内容，选择「文件导入」
""",
    openapi_tags=[
        {"name": "查询", "description": "自然语言转 SQL 查询接口"},
        {"name": "系统", "description": "系统运维与监控接口"},
    ],
)
system = ChatBISystem()

# ==================== CORS 中间件 ====================
# 前后端分离时，浏览器从 localhost:3000/file:// 访问 localhost:8000 会触发跨域限制
# CORS 中间件允许跨域请求，是前后端联调的必要配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """查询请求体"""

    question: str = Field(..., min_length=1, description="业务人员的自然语言问题")
    use_few_shot: bool = Field(default=True, description="是否启用 Few-shot 示例")
    use_rules: bool = Field(default=True, description="是否启用业务规则约束")
    use_guards: bool = Field(default=True, description="是否启用错误防护")
    use_indicator_knowledge: bool = Field(default=True, description="是否注入指标知识")
    use_schema_linking: bool = Field(default=False, description="是否启用 Schema Linking")
    use_indicator_rag: bool = Field(default=False, description="是否启用指标 RAG")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    database_connected: bool


class QuerySuccessResponse(BaseModel):
    """成功响应"""

    success: bool = True
    question: str
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    formatted: str
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    """错误响应"""

    success: bool = False
    error: str
    error_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _rows_to_dicts(columns: list[str], results: list[tuple]) -> list[dict[str, Any]]:
    """将数据库元组结果转换为 JSON 可直接返回的字典列表"""
    return [dict(zip(columns, row)) for row in results]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """统一处理请求体验证错误"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="请求参数校验失败",
            error_type="request_validation",
            metadata={
                "path": str(request.url.path),
                "details": exc.errors(),
            },
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """统一处理业务异常"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            error_type="http_exception",
            metadata={"path": str(request.url.path)},
        ).model_dump()
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底异常处理，避免把 Python Traceback 直接暴露给前端"""
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="服务内部异常",
            error_type="internal_server_error",
            metadata={"path": str(request.url.path)},
        ).model_dump()
    )


@app.get("/", tags=["系统"])
def read_root() -> dict[str, str]:
    """服务说明入口"""
    return {
        "name": "ChatBI MVP API",
        "docs": "/docs",
        "health": "/health",
        "query": "/api/v1/query",
        "query_stream": "/api/v1/query/stream",
    }


@app.get("/health", response_model=HealthResponse, tags=["系统"])
def health_check() -> HealthResponse:
    """检查 API 服务和数据库连通性"""
    return HealthResponse(
        status="ok",
        database_connected=system.db.validate_connection()
    )


@app.post(
    "/api/v1/query",
    response_model=QuerySuccessResponse,
    tags=["查询"],
    summary="同步查询（一次性返回）",
    responses={
        400: {"model": ErrorResponse, "description": "输入问题不合法"},
        502: {"model": ErrorResponse, "description": "LLM 调用失败"},
        500: {"model": ErrorResponse, "description": "数据库或服务内部异常"},
    },
)
def query_chatbi(payload: QueryRequest) -> QuerySuccessResponse:
    """执行自然语言查询，并返回标准化结果（同步，一次性返回）"""
    started_at = perf_counter()
    logger.info("Received question: %s", payload.question)

    result = system.run(
        user_question=payload.question,
        use_few_shot=payload.use_few_shot,
        use_rules=payload.use_rules,
        use_guards=payload.use_guards,
        use_indicator_knowledge=payload.use_indicator_knowledge,
        use_schema_linking=payload.use_schema_linking,
        use_indicator_rag=payload.use_indicator_rag,
    )

    duration_ms = round((perf_counter() - started_at) * 1000, 2)

    if not result["success"]:
        error_type = result.get("error_type", "internal_server_error")
        status_code = 500
        if error_type == "validation":
            status_code = 400
        elif error_type == "llm":
            status_code = 502
        raise HTTPException(status_code=status_code, detail=result["error"])

    metadata = {
        **result.get("metadata", {}),
        "duration_ms": duration_ms,
    }
    logger.info("Question handled successfully in %.2f ms", duration_ms)

    return QuerySuccessResponse(
        question=payload.question,
        sql=result["sql"],
        columns=result["columns"],
        rows=_rows_to_dicts(result["columns"], result["results"]),
        formatted=result["formatted"],
        metadata=metadata,
    )


@app.post("/api/v1/query/stream", tags=["查询"], summary="SSE 流式查询（逐步推送）")
async def query_chatbi_stream(payload: QueryRequest) -> StreamingResponse:
    """
    执行自然语言查询，以 SSE 流式返回结果

    SSE 事件类型：
    - sql_chunk: LLM 流式产出的 SQL 文本片段，前端可逐字拼接展示
    - sql_done: SQL 完整输出，前端可用于复制或二次处理
    - result: 查询结果（columns + rows）
    - error: 异常信息

    每个事件的格式为：
        event: <type>
        data: <json>
    """
    logger.info("Stream request received: %s", payload.question)

    def event_generator():
        for event_str in system.run_stream(
            user_question=payload.question,
            use_few_shot=payload.use_few_shot,
            use_rules=payload.use_rules,
            use_guards=payload.use_guards,
            use_indicator_knowledge=payload.use_indicator_knowledge,
            use_schema_linking=payload.use_schema_linking,
            use_indicator_rag=payload.use_indicator_rag,
        ):
            yield event_str

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 环境下禁止缓冲
        },
    )


# ==================== 静态文件挂载 ====================
# 将 static/ 目录挂载为 /static 路径，前端 HTML 页面可直接访问
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
