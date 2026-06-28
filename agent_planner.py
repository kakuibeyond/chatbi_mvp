"""
Plan-and-Execute Agent 骨架模块

承接第 22 课的 Query 拆解结果，补齐 Planner、Executor、Summarizer 三个角色，
形成“复杂问题 -> 子任务 -> 执行计划 -> 多步执行 -> 结果汇总”的最小闭环。
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from typing import Any, Callable

from pydantic import BaseModel, Field

from main import ChatBISystem
from query_decomposer import DecompositionPlan, DecomposedTask, QueryDecomposer


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
    question: str
    depends_on: list[str] = Field(default_factory=list)
    context_used: str = ""
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


class StepExecutor:
    """顺序执行计划中的每个步骤。"""

    def __init__(
        self,
        step_runner: StepRunner | None = None,
        chatbi_system: ChatBISystem | None = None,
        chatbi_run_options: dict[str, Any] | None = None,
    ):
        self.system = chatbi_system
        self.chatbi_run_options = chatbi_run_options or {
            "use_schema_linking": True,
            "use_indicator_rag": True,
            "use_indicator_knowledge": True,
        }
        self.step_runner = step_runner or self._run_with_chatbi

    def execute_plan(
        self,
        plan: ExecutionPlan,
        max_steps: int | None = None,
    ) -> list[StepExecutionResult]:
        results: list[StepExecutionResult] = []
        results_by_step: dict[str, StepExecutionResult] = {}
        steps_to_run = plan.steps[:max_steps] if max_steps is not None else plan.steps

        for step in steps_to_run:
            dependency_context = self._build_dependency_context(
                step.depends_on,
                results_by_step,
            )
            composed_question = self._compose_question(step.question, dependency_context)
            raw_result = self.step_runner(composed_question)
            normalized = self._normalize_result(
                step=step,
                question=composed_question,
                context_used=dependency_context,
                raw_result=raw_result,
            )
            results.append(normalized)
            results_by_step[step.step_id] = normalized

        return results

    def _run_with_chatbi(self, question: str) -> dict[str, Any]:
        if self.system is None:
            self.system = ChatBISystem()

        return self.system.run(
            user_question=question,
            **self.chatbi_run_options,
        )

    @staticmethod
    def _compose_question(step_question: str, dependency_context: str) -> str:
        if not dependency_context:
            return step_question
        return (
            f"{step_question}\n\n"
            f"前置步骤关键结果如下，请在本次查询中延续这些上下文：\n"
            f"{dependency_context}"
        )

    @staticmethod
    def _build_dependency_context(
        dependency_ids: list[str],
        results_by_step: dict[str, StepExecutionResult],
    ) -> str:
        if not dependency_ids:
            return ""

        lines: list[str] = []
        for step_id in dependency_ids:
            result = results_by_step[step_id]
            lines.append(
                f"- {result.step_name}：{StepExecutor._pick_result_brief(result)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _pick_result_brief(result: StepExecutionResult) -> str:
        if not result.success:
            return f"执行失败，错误信息：{result.error or '未知错误'}"

        if result.formatted.strip():
            first_line = result.formatted.strip().splitlines()[0]
            return first_line[:120]

        if result.rows:
            return json.dumps(result.rows[0], ensure_ascii=False)

        return "步骤执行成功，但当前无返回行。"

    @staticmethod
    def _normalize_result(
        step: PlanStep,
        question: str,
        context_used: str,
        raw_result: dict[str, Any],
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
            question=question,
            depends_on=step.depends_on,
            context_used=context_used,
            sql=raw_result.get("sql"),
            columns=columns,
            rows=normalized_rows,
            formatted=raw_result.get("formatted", ""),
            error=raw_result.get("error"),
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
        failed = [result for result in step_results if not result.success]

        findings = [
            f"{result.step_name}：{StepExecutor._pick_result_brief(result)}"
            for result in step_results
        ]

        if failed:
            summary_text = (
                f"已完成 {len(completed)} 个步骤，"
                f"失败 {len(failed)} 个步骤。"
                "当前链路已经暴露出真实执行问题，"
                "需要先修复失败步骤，再继续扩展中间结果管理与总结能力。"
            )
        else:
            summary_text = (
                f"已完成 {len(completed)} 个步骤，"
                f"失败 {len(failed)} 个步骤。"
                "当前结果已经可以支撑后续的中间结果管理与最终报告生成。"
            )

        return ExecutionSummary(
            original_question=original_question,
            analysis_goal=plan.analysis_goal,
            completed_steps=len(completed),
            failed_steps=len(failed),
            key_findings=findings,
            summary_text=summary_text,
        )


class PlanAndExecuteAgent:
    """Plan-and-Execute Agent 总入口。"""

    def __init__(
        self,
        decomposer: QueryDecomposer | None = None,
        planner: PlanGenerator | None = None,
        executor: StepExecutor | None = None,
        summarizer: ResultSummarizer | None = None,
    ):
        self.decomposer = decomposer or QueryDecomposer()
        self.planner = planner or PlanGenerator()
        self.executor = executor or StepExecutor()
        self.summarizer = summarizer or ResultSummarizer()

    def run(
        self,
        user_question: str,
        decomposition_override: dict[str, Any] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        decomposition = decomposition_override or self.decomposer.decompose(user_question)
        plan = self.planner.build_plan(user_question, decomposition)
        step_results = self.executor.execute_plan(plan, max_steps=max_steps)
        summary = self.summarizer.summarize(user_question, plan, step_results)

        return {
            "original_question": user_question,
            "decomposition": decomposition,
            "plan": plan.model_dump(),
            "step_results": [result.model_dump() for result in step_results],
            "summary": summary.model_dump(),
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
    args = parser.parse_args()

    if args.plan_only:
        decomposer = QueryDecomposer()
        planner = PlanGenerator()
        decomposition = decomposer.decompose(args.question)
        plan = planner.build_plan(args.question, decomposition)
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
            }
        )
        agent = PlanAndExecuteAgent(executor=executor)
        result = agent.run(args.question, max_steps=args.max_steps)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
