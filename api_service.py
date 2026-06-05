"""
FastAPI 服务入口

将命令行版 ChatBI 系统封装为 REST API，供前端或其他服务调用。
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from main import ChatBISystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("chatbi.api")

app = FastAPI(
    title="ChatBI MVP API",
    version="0.1.0",
    description="企业级 ChatBI MVP 的服务化接口。"
)
system = ChatBISystem()


class QueryRequest(BaseModel):
    """查询请求体"""

    question: str = Field(..., min_length=1, description="业务人员的自然语言问题")
    use_few_shot: bool = Field(default=True, description="是否启用 Few-shot 示例")
    use_rules: bool = Field(default=True, description="是否启用业务规则约束")
    use_guards: bool = Field(default=True, description="是否启用错误防护")
    use_indicator_knowledge: bool = Field(default=True, description="是否注入指标知识")


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


@app.get("/")
def read_root() -> dict[str, str]:
    """服务说明入口"""
    return {
        "name": "ChatBI MVP API",
        "docs": "/docs",
        "health": "/health",
        "query": "/api/v1/query",
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """检查 API 服务和数据库连通性"""
    return HealthResponse(
        status="ok",
        database_connected=system.db.validate_connection()
    )


@app.post(
    "/api/v1/query",
    response_model=QuerySuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "输入问题不合法"},
        502: {"model": ErrorResponse, "description": "LLM 调用失败"},
        500: {"model": ErrorResponse, "description": "数据库或服务内部异常"},
    },
)
def query_chatbi(payload: QueryRequest) -> QuerySuccessResponse:
    """执行自然语言查询，并返回标准化结果"""
    started_at = perf_counter()
    logger.info("Received question: %s", payload.question)

    result = system.run(
        user_question=payload.question,
        use_few_shot=payload.use_few_shot,
        use_rules=payload.use_rules,
        use_guards=payload.use_guards,
        use_indicator_knowledge=payload.use_indicator_knowledge,
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


if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
