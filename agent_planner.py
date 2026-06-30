"""
Plan-and-Execute Agent 骨架模块

承接第 22 课的 Query 拆解结果，补齐 Planner、Executor、Summarizer 三个角色，
形成“复杂问题 -> 子任务 -> 执行计划 -> 多步执行 -> 结果汇总”的最小闭环。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Literal

import pymysql
from pydantic import BaseModel, Field

from config import DB_CONFIG
from main import ChatBISystem
from query_decomposer import DecompositionPlan, DecomposedTask, QueryDecomposer
from report_generator import ReportGenerator


class PlanStep(BaseModel):
    """单个执行步骤定义。"""

    step_id: str
    task_id: str
    step_name: str
    task_type: str
    action: str = "text2sql"
    question: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    expected_output: str


class ExecutionPlan(BaseModel):
    """可执行计划。"""

    original_question: str
    question_type: str
    analysis_goal: str
    steps: list[PlanStep]


class StepExecutionResult(BaseModel):
    """单步执行结果。"""

    step_id: str
    task_id: str
    step_name: str
    success: bool
    status: Literal["completed", "failed", "skipped"] = "completed"
    attempts: int = 1
    question: str
    depends_on: list[str] = Field(default_factory=list)
    context_used: str = ""
    storage_backend: str | None = None
    result_reference: str | None = None
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    formatted: str = ""
    error: str | None = None


class ExecutionSummary(BaseModel):
    """执行摘要。"""

    original_question: str
    analysis_goal: str
    completed_steps: int
    failed_steps: int
    skipped_steps: int = 0
    key_findings: list[str] = Field(default_factory=list)
    summary_text: str


class PlanGenerator:
    """根据拆解结果生成可执行计划。"""

    def build_plan(
        self,
        original_question: str,
        decomposition: dict[str, Any] | DecompositionPlan,
    ) -> ExecutionPlan:
        plan = self._ensure_decomposition_plan(decomposition)
        task_to_step = {
            task.task_id: f"step_{index}"
            for index, task in enumerate(plan.subtasks, start=1)
        }

        steps: list[PlanStep] = []
        for index, task in enumerate(plan.subtasks, start=1):
            steps.append(
                PlanStep(
                    step_id=f"step_{index}",
                    task_id=task.task_id,
                    step_name=task.task_name,
                    task_type=task.task_type,
                    question=self._build_step_question(task),
                    description=task.description,
                    depends_on=[task_to_step[dep] for dep in task.depends_on],
                    metrics=task.metrics,
                    dimensions=task.dimensions,
                    expected_output=self._build_expected_output(task),
                )
            )

        return ExecutionPlan(
            original_question=original_question,
            question_type=plan.question_type,
            analysis_goal=plan.analysis_goal,
            steps=steps,
        )

    @staticmethod
    def _ensure_decomposition_plan(
        decomposition: dict[str, Any] | DecompositionPlan,
    ) -> DecompositionPlan:
        if isinstance(decomposition, DecompositionPlan):
            return decomposition
        return DecompositionPlan.model_validate(decomposition)

    @staticmethod
    def _build_step_question(task: DecomposedTask) -> str:
        lines = [f"请执行子任务：{task.task_name}。"]
        if task.description:
            lines.append(f"任务说明：{task.description}")
        if task.metrics:
            lines.append(f"关注指标：{'、'.join(task.metrics)}")
        if task.dimensions:
            lines.append(f"分析维度：{'、'.join(task.dimensions)}")
        lines.append("执行约束：本步骤只回答当前子任务，优先生成一条结构清晰、易执行的 SQL。")
        lines.append("请优先返回支撑后续分析的查询结果，不要直接跳到最终归因结论。")
        return "\n".join(lines)

    @staticmethod
    def _build_expected_output(task: DecomposedTask) -> str:
        focus_parts: list[str] = []
        if task.metrics:
            focus_parts.append(f"指标：{'、'.join(task.metrics)}")
        if task.dimensions:
            focus_parts.append(f"维度：{'、'.join(task.dimensions)}")
        if not focus_parts:
            focus_parts.append("可被后续步骤复用的结构化结果")
        return "；".join(focus_parts)


StepRunner = Callable[[str], dict[str, Any]]

def _print_progress(message: str) -> None:
    """直接打印主链路进度日志。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


class IntermediateResultStore:
    """中间结果存储抽象。"""

    def put(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    def get(self, reference: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def cleanup(self) -> None:
        """释放临时资源。"""


class MemoryResultStore(IntermediateResultStore):
    """基于进程内字典的轻量存储。"""

    def __init__(self):
        self.data: dict[str, list[dict[str, Any]]] = {}

    def put(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        reference = f"memory://{step_id}"
        self.data[reference] = rows
        return reference

    def get(self, reference: str) -> list[dict[str, Any]]:
        return self.data.get(reference, [])

    def cleanup(self) -> None:
        self.data.clear()


class TempTableResultStore(IntermediateResultStore):
    """基于 MySQL 临时表的中间结果存储。"""

    def __init__(
        self,
        connection_factory: Callable[[], Any] | None = None,
    ):
        self.connection_factory = connection_factory or (
            lambda: pymysql.connect(**DB_CONFIG)
        )
        self.connection: Any | None = None
        self.reference_to_table: dict[str, str] = {}

    def put(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        conn = self._ensure_connection()
        table_name = f"tmp_agent_{self._sanitize_identifier(step_id)}"
        table_columns = columns or (list(rows[0].keys()) if rows else ["_empty_marker"])
        column_defs = ", ".join(
            f"{self._quote_identifier(column)} {self._infer_sql_type([row.get(column) for row in rows])}"
            for column in table_columns
        )

        with conn.cursor() as cursor:
            cursor.execute(
                f"DROP TEMPORARY TABLE IF EXISTS {self._quote_identifier(table_name)}"
            )
            cursor.execute(
                f"CREATE TEMPORARY TABLE {self._quote_identifier(table_name)} ({column_defs})"
            )
            if rows:
                placeholders = ", ".join(["%s"] * len(table_columns))
                column_names = ", ".join(
                    self._quote_identifier(column) for column in table_columns
                )
                values = [
                    tuple(row.get(column) for column in table_columns)
                    for row in rows
                ]
                cursor.executemany(
                    f"INSERT INTO {self._quote_identifier(table_name)} ({column_names}) VALUES ({placeholders})",
                    values,
                )
        conn.commit()

        reference = f"temp_table://{table_name}"
        self.reference_to_table[reference] = table_name
        return reference

    def get(self, reference: str) -> list[dict[str, Any]]:
        conn = self._ensure_connection()
        table_name = self.reference_to_table.get(reference) or self._parse_reference(reference)

        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {self._quote_identifier(table_name)}")
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def cleanup(self) -> None:
        if self.connection is None:
            return

        try:
            with self.connection.cursor() as cursor:
                for table_name in set(self.reference_to_table.values()):
                    cursor.execute(
                        f"DROP TEMPORARY TABLE IF EXISTS {self._quote_identifier(table_name)}"
                    )
            self.connection.commit()
        finally:
            self.connection.close()
            self.connection = None
            self.reference_to_table.clear()

    def _ensure_connection(self) -> Any:
        if self.connection is None or not getattr(self.connection, "open", True):
            self.connection = self.connection_factory()
        return self.connection

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", value)
        return sanitized.strip("_") or "step"

    @staticmethod
    def _quote_identifier(value: str) -> str:
        escaped = value.replace("`", "``")
        return f"`{escaped}`"

    @staticmethod
    def _infer_sql_type(values: list[Any]) -> str:
        non_null = next((value for value in values if value is not None), None)
        if isinstance(non_null, bool):
            return "TINYINT(1)"
        if isinstance(non_null, int):
            return "BIGINT"
        if isinstance(non_null, (float, Decimal)):
            return "DOUBLE"
        return "TEXT"

    @staticmethod
    def _parse_reference(reference: str) -> str:
        if not reference.startswith("temp_table://"):
            raise ValueError(f"非法的临时表引用：{reference}")
        return reference.split("://", 1)[1]


class StepExecutor:
    """顺序执行计划中的每个步骤。"""

    def __init__(
        self,
        step_runner: StepRunner | None = None,
        chatbi_system: ChatBISystem | None = None,
        chatbi_run_options: dict[str, Any] | None = None,
        max_retries: int = 0,
        failure_policy: Literal["abort", "skip"] = "abort",
        storage_backend: Literal["memory", "temp_table"] = "memory",
        result_store: IntermediateResultStore | None = None,
        storage_connection_factory: Callable[[], Any] | None = None,
    ):
        self.system = chatbi_system
        self.chatbi_run_options = chatbi_run_options or {
            "use_schema_linking": True,
            "use_indicator_rag": True,
            "use_indicator_knowledge": True,
        }
        self.step_runner = step_runner or self._run_with_chatbi
        self.max_retries = max_retries
        self.failure_policy = failure_policy
        self.storage_backend = storage_backend
        self.result_store = result_store or self._build_result_store(
            storage_backend=storage_backend,
            connection_factory=storage_connection_factory,
        )
        self.memory_store = (
            self.result_store.data
            if isinstance(self.result_store, MemoryResultStore)
            else {}
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        max_steps: int | None = None,
    ) -> list[StepExecutionResult]:
        results: list[StepExecutionResult] = []
        results_by_step: dict[str, StepExecutionResult] = {}
        steps_to_run = plan.steps[:max_steps] if max_steps is not None else plan.steps
        abort_triggered = False
        total_steps = len(steps_to_run)

        _print_progress(f"开始执行计划，共 {total_steps} 个步骤。")

        for index, step in enumerate(steps_to_run, start=1):
            if abort_triggered:
                _print_progress(
                    f"{step.step_id} 已跳过：前序步骤失败，当前执行策略为 abort。"
                )
                skipped = self._build_skipped_result(
                    step=step,
                    error="前序步骤失败，当前执行策略为 abort，后续步骤停止执行。",
                )
                results.append(skipped)
                results_by_step[step.step_id] = skipped
                continue

            if self._has_failed_dependency(step.depends_on, results_by_step):
                _print_progress(
                    f"{step.step_id} 已跳过：依赖步骤失败。"
                )
                skipped = self._build_skipped_result(
                    step=step,
                    error="依赖步骤失败，当前步骤已跳过。",
                )
                results.append(skipped)
                results_by_step[step.step_id] = skipped
                continue

            _print_progress(
                f"开始执行 {step.step_id}（{index}/{total_steps}）：{step.step_name}"
            )
            dependency_context = self._build_dependency_context(
                step.depends_on,
                results_by_step,
            )
            composed_question = self._compose_question(step.question, dependency_context)
            normalized = self._execute_with_retry(
                step=step,
                question=composed_question,
                context_used=dependency_context,
            )
            if normalized.success:
                normalized.result_reference = self._store_intermediate_result(
                    step.step_id,
                    normalized.columns,
                    normalized.rows,
                )
                normalized.storage_backend = self.storage_backend
                _print_progress(
                    f"{step.step_id} 执行完成：{step.step_name}"
                )
            elif self.failure_policy == "abort":
                _print_progress(
                    f"{step.step_id} 执行失败：{normalized.error or '未知错误'}"
                )
                abort_triggered = True
            else:
                _print_progress(
                    f"{step.step_id} 执行失败：{normalized.error or '未知错误'}"
                )

            results.append(normalized)
            results_by_step[step.step_id] = normalized

        completed_steps = sum(1 for result in results if result.status == "completed")
        failed_steps = sum(1 for result in results if result.status == "failed")
        skipped_steps = sum(1 for result in results if result.status == "skipped")
        _print_progress(
            f"执行计划结束：completed={completed_steps}, failed={failed_steps}, skipped={skipped_steps}"
        )
        return results

    def _run_with_chatbi(self, question: str) -> dict[str, Any]:
        if self.system is None:
            self.system = ChatBISystem()

        return self.system.run(
            user_question=question,
            **self.chatbi_run_options,
        )

    def _execute_with_retry(
        self,
        step: PlanStep,
        question: str,
        context_used: str,
    ) -> StepExecutionResult:
        last_result: StepExecutionResult | None = None

        for attempt in range(1, self.max_retries + 2):
            if attempt > 1:
                _print_progress(
                    f"{step.step_id} 开始第 {attempt} 次尝试：{step.step_name}"
                )
            raw_result = self.step_runner(question)
            normalized = self._normalize_result(
                step=step,
                question=question,
                context_used=context_used,
                raw_result=raw_result,
                attempts=attempt,
                storage_backend=self.storage_backend,
            )
            if normalized.success:
                return normalized
            last_result = normalized

        assert last_result is not None
        return last_result

    @staticmethod
    def _compose_question(step_question: str, dependency_context: str) -> str:
        if not dependency_context:
            return step_question
        return (
            f"{step_question}\n\n"
            f"前置步骤关键结果如下，请在本次查询中延续这些上下文：\n"
            f"{dependency_context}"
        )

    def _build_dependency_context(
        self,
        dependency_ids: list[str],
        results_by_step: dict[str, StepExecutionResult],
    ) -> str:
        if not dependency_ids:
            return ""

        lines: list[str] = []
        for step_id in dependency_ids:
            result = results_by_step[step_id]
            rows = result.rows or self.get_intermediate_result(result.result_reference)
            payload = {
                "step_id": result.step_id,
                "step_name": result.step_name,
                "columns": result.columns,
                "rows": rows,
                "result_reference": result.result_reference,
            }
            lines.append(json.dumps(payload, ensure_ascii=False, default=str))
        return "\n".join(lines)

    @staticmethod
    def _has_failed_dependency(
        dependency_ids: list[str],
        results_by_step: dict[str, StepExecutionResult],
    ) -> bool:
        return any(not results_by_step[step_id].success for step_id in dependency_ids)

    def _pick_result_brief(self, result: StepExecutionResult) -> str:
        if not result.success:
            return f"执行失败，错误信息：{result.error or '未知错误'}"

        if result.formatted.strip() and not self._looks_like_table(result.formatted):
            meaningful_line = self._pick_meaningful_formatted_line(result.formatted)
            if meaningful_line:
                return meaningful_line

        if result.rows:
            return self._summarize_rows(result.rows)

        if result.formatted.strip():
            meaningful_line = self._pick_meaningful_formatted_line(result.formatted)
            if meaningful_line:
                return meaningful_line

        stored_rows = self.get_intermediate_result(result.result_reference)
        if stored_rows:
            return json.dumps(stored_rows[0], ensure_ascii=False)

        if result.rows:
            return json.dumps(result.rows[0], ensure_ascii=False)

        return "步骤执行成功，但当前无返回行。"

    @staticmethod
    def _summarize_rows(rows: list[dict[str, Any]]) -> str:
        first_row = rows[0]
        parts = [
            f"{key}={value}"
            for key, value in first_row.items()
            if value is not None
        ]
        return "，".join(parts[:4])[:120]

    @staticmethod
    def _pick_meaningful_formatted_line(formatted: str) -> str:
        for line in formatted.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if set(stripped) <= {"-", "+"}:
                continue
            return stripped[:120]
        return ""

    @staticmethod
    def _looks_like_table(formatted: str) -> bool:
        text = formatted.strip()
        return "+" in text and "|" in text

    @staticmethod
    def _normalize_result(
        step: PlanStep,
        question: str,
        context_used: str,
        raw_result: dict[str, Any],
        attempts: int = 1,
        storage_backend: str | None = None,
    ) -> StepExecutionResult:
        columns = raw_result.get("columns", [])
        rows = raw_result.get("results") or raw_result.get("rows") or []

        if rows and columns and isinstance(rows[0], tuple):
            normalized_rows = [dict(zip(columns, row)) for row in rows]
        else:
            normalized_rows = rows

        return StepExecutionResult(
            step_id=step.step_id,
            task_id=step.task_id,
            step_name=step.step_name,
            success=raw_result.get("success", False),
            status="completed" if raw_result.get("success", False) else "failed",
            attempts=attempts,
            question=question,
            depends_on=step.depends_on,
            context_used=context_used,
            storage_backend=storage_backend,
            sql=raw_result.get("sql"),
            columns=columns,
            rows=normalized_rows,
            formatted=raw_result.get("formatted", ""),
            error=raw_result.get("error"),
        )

    def _store_intermediate_result(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        return self.result_store.put(step_id=step_id, columns=columns, rows=rows)

    def get_intermediate_result(
        self,
        reference: str | None,
    ) -> list[dict[str, Any]]:
        if not reference:
            return []
        return self.result_store.get(reference)

    def cleanup(self) -> None:
        self.result_store.cleanup()

    @staticmethod
    def _build_result_store(
        storage_backend: Literal["memory", "temp_table"],
        connection_factory: Callable[[], Any] | None = None,
    ) -> IntermediateResultStore:
        if storage_backend == "temp_table":
            return TempTableResultStore(connection_factory=connection_factory)
        return MemoryResultStore()

    @staticmethod
    def _build_skipped_result(step: PlanStep, error: str) -> StepExecutionResult:
        return StepExecutionResult(
            step_id=step.step_id,
            task_id=step.task_id,
            step_name=step.step_name,
            success=False,
            status="skipped",
            question=step.question,
            depends_on=step.depends_on,
            error=error,
        )


class ResultSummarizer:
    """把多步执行结果收敛为统一摘要。"""

    def summarize(
        self,
        original_question: str,
        plan: ExecutionPlan,
        step_results: list[StepExecutionResult],
    ) -> ExecutionSummary:
        completed = [result for result in step_results if result.success]
        failed = [result for result in step_results if result.status == "failed"]
        skipped = [result for result in step_results if result.status == "skipped"]

        findings = [
            f"{result.step_name}：{self._pick_result_brief(result)}"
            for result in step_results
        ]

        if failed:
            summary_text = (
                f"已完成 {len(completed)} 个步骤，"
                f"失败 {len(failed)} 个步骤，"
                f"跳过 {len(skipped)} 个步骤。"
                "当前链路已经暴露出真实执行问题，"
                "需要先修复失败步骤，再继续扩展中间结果管理与总结能力。"
            )
        else:
            summary_text = (
                f"已完成 {len(completed)} 个步骤，"
                f"失败 {len(failed)} 个步骤，"
                f"跳过 {len(skipped)} 个步骤。"
                "当前结果已经可以支撑后续的中间结果管理与最终报告生成。"
            )

        return ExecutionSummary(
            original_question=original_question,
            analysis_goal=plan.analysis_goal,
            completed_steps=len(completed),
            failed_steps=len(failed),
            skipped_steps=len(skipped),
            key_findings=findings,
            summary_text=summary_text,
        )

    @staticmethod
    def _pick_result_brief(result: StepExecutionResult) -> str:
        if not result.success:
            return f"执行失败，错误信息：{result.error or '未知错误'}"
        if result.formatted.strip() and not ResultSummarizer._looks_like_table(result.formatted):
            for line in result.formatted.strip().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if set(stripped) <= {"-", "+"}:
                    continue
                return stripped[:120]
        if result.rows:
            first_row = result.rows[0]
            parts = [
                f"{key}={value}"
                for key, value in first_row.items()
                if value is not None
            ]
            return "，".join(parts[:4])[:120]
        if result.formatted.strip():
            for line in result.formatted.strip().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if set(stripped) <= {"-", "+"}:
                    continue
                return stripped[:120]
        if result.rows:
            return json.dumps(result.rows[0], ensure_ascii=False)
        if result.result_reference:
            return f"结果已写入 {result.result_reference}"
        return "步骤执行成功，但当前无返回行。"

    @staticmethod
    def _looks_like_table(formatted: str) -> bool:
        text = formatted.strip()
        return "+" in text and "|" in text

class PlanAndExecuteAgent:
    """Plan-and-Execute Agent 总入口。"""

    def __init__(
        self,
        decomposer: QueryDecomposer | None = None,
        planner: PlanGenerator | None = None,
        executor: StepExecutor | None = None,
        summarizer: ResultSummarizer | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.decomposer = decomposer or QueryDecomposer()
        self.planner = planner or PlanGenerator()
        self.executor = executor or StepExecutor()
        self.summarizer = summarizer or ResultSummarizer()
        self.report_generator = report_generator or ReportGenerator()

    def run(
        self,
        user_question: str,
        decomposition_override: dict[str, Any] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        if decomposition_override is not None:
            _print_progress("使用传入的拆解结果，跳过任务拆解。")
            decomposition = decomposition_override
        else:
            _print_progress("开始任务拆解。")
            decomposition = self.decomposer.decompose(user_question)
            _print_progress(
                f"任务拆解完成，共 {len(decomposition.get('subtasks', []))} 个子任务。"
            )

        _print_progress("开始生成执行计划。")
        plan = self.planner.build_plan(user_question, decomposition)
        _print_progress(f"执行计划生成完成，共 {len(plan.steps)} 个步骤。")
        step_results = self.executor.execute_plan(plan, max_steps=max_steps)
        _print_progress("开始生成执行摘要。")
        summary = self.summarizer.summarize(user_question, plan, step_results)
        _print_progress("执行摘要生成完成。")
        _print_progress("开始生成分析报告。")
        report = self.report_generator.generate(
            original_question=user_question,
            analysis_goal=plan.analysis_goal,
            step_results=[result.model_dump() for result in step_results],
            summary=summary.model_dump(),
        )
        _print_progress("分析报告生成完成。")

        return {
            "original_question": user_question,
            "decomposition": decomposition,
            "plan": plan.model_dump(),
            "step_results": [result.model_dump() for result in step_results],
            "summary": summary.model_dump(),
            "report": report.model_dump(),
        }


def _json_default(value: Any) -> str:
    """为 CLI 输出提供兜底序列化。"""
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-and-Execute Agent 骨架")
    parser.add_argument("question", nargs="?", default="最近三个月利润为什么下降？")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只执行真实拆解与计划生成，不执行后续查询。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="限制实际执行的步骤数，便于分步验证真实链路。",
    )
    parser.add_argument(
        "--disable-schema-linking",
        action="store_true",
        help="执行步骤时关闭 Schema Linking，直接使用基础 Text2SQL 链路。",
    )
    parser.add_argument(
        "--disable-indicator-rag",
        action="store_true",
        help="执行步骤时关闭指标 RAG，避免额外检索链路干扰验证。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="单步执行失败后的最大重试次数。",
    )
    parser.add_argument(
        "--failure-policy",
        choices=["abort", "skip"],
        default="abort",
        help="步骤失败后如何处理后续链路。",
    )
    parser.add_argument(
        "--storage-backend",
        choices=["memory", "temp_table"],
        default="memory",
        help="中间结果引用方式。",
    )
    args = parser.parse_args()

    if args.plan_only:
        _print_progress("开始任务拆解。")
        decomposer = QueryDecomposer()
        planner = PlanGenerator()
        decomposition = decomposer.decompose(args.question)
        _print_progress(
            f"任务拆解完成，共 {len(decomposition.get('subtasks', []))} 个子任务。"
        )
        _print_progress("开始生成执行计划。")
        plan = planner.build_plan(args.question, decomposition)
        _print_progress(f"执行计划生成完成，共 {len(plan.steps)} 个步骤。")
        result = {
            "original_question": args.question,
            "decomposition": decomposition,
            "plan": plan.model_dump(),
        }
    else:
        executor = StepExecutor(
            chatbi_run_options={
                "use_schema_linking": not args.disable_schema_linking,
                "use_indicator_rag": not args.disable_indicator_rag,
                "use_indicator_knowledge": True,
            },
            max_retries=args.max_retries,
            failure_policy=args.failure_policy,
            storage_backend=args.storage_backend,
        )
        agent = PlanAndExecuteAgent(executor=executor)
        result = agent.run(args.question, max_steps=args.max_steps)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
