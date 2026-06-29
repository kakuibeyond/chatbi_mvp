"""
分析报告生成模块

把多步 SQL 执行后的结构化结果，整理为业务可读的分析报告。
默认优先调用 LLM 输出结构化 JSON；如果模型不可用或返回格式异常，
则回退到确定性的模板化报告，保证 Agent 输出层始终可用。
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Callable

from pydantic import BaseModel, Field

from config import LLM_CONFIG
from llm_client import LLMClient


ReportTextGenerator = Callable[[str, str], str]


class AnalysisReport(BaseModel):
    """面向业务读者的结构化分析报告。"""

    title: str
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    trend_judgment: str
    action_suggestions: list[str] = Field(default_factory=list)
    markdown: str = ""


class ReportGenerator:
    """将执行结果收敛为结构化报告。"""

    def __init__(self, text_generator: ReportTextGenerator | None = None):
        self.text_generator = text_generator or self._build_default_text_generator()

    def generate(
        self,
        original_question: str,
        analysis_goal: str,
        step_results: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> AnalysisReport:
        if self.text_generator is None:
            return self._build_fallback_report(
                original_question=original_question,
                analysis_goal=analysis_goal,
                step_results=step_results,
                summary=summary,
            )

        system_msg = self._build_system_message()
        prompt = self._build_prompt(
            original_question=original_question,
            analysis_goal=analysis_goal,
            step_results=step_results,
            summary=summary,
        )

        try:
            raw_text = self.text_generator(system_msg, prompt)
            parsed = self._parse_report_json(raw_text)
            parsed.markdown = self._render_markdown(parsed)
            return parsed
        except Exception:
            return self._build_fallback_report(
                original_question=original_question,
                analysis_goal=analysis_goal,
                step_results=step_results,
                summary=summary,
            )

    @staticmethod
    def _build_system_message() -> str:
        return (
            "你是企业经营分析师，负责把 SQL 查询结果整理成业务人员可读的分析报告。"
            "只基于提供的数据和执行结果写结论，不要编造未出现的事实。"
            "输出必须是 JSON。"
        )

    def _build_prompt(
        self,
        original_question: str,
        analysis_goal: str,
        step_results: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> str:
        report_context = {
            "original_question": original_question,
            "analysis_goal": analysis_goal,
            "summary": summary,
            "step_results": [self._compress_step_result(step_result) for step_result in step_results],
        }
        context_json = json.dumps(
            report_context,
            ensure_ascii=False,
            indent=2,
            default=self._json_default,
        )
        return (
            "请根据下面的 Agent 执行结果，生成一份结构化分析报告。\n"
            "报告应包含：标题、执行摘要、关键发现、归因分析、趋势判断、行动建议。\n"
            "要求：\n"
            "1. 只使用上下文里已经出现的信息。\n"
            "2. 如果数据不足，结论要保守。\n"
            "3. 关键发现和行动建议用简短句子表达。\n"
            "4. 只返回 JSON，不要额外输出 Markdown。\n"
            "5. JSON 必须包含 title、executive_summary、key_findings、root_causes、trend_judgment、action_suggestions 这 6 个字段。\n\n"
            f"上下文如下：\n{context_json}"
        )

    @staticmethod
    def _compress_step_result(step_result: dict[str, Any]) -> dict[str, Any]:
        rows = step_result.get("rows") or []
        return {
            "step_id": step_result.get("step_id"),
            "step_name": step_result.get("step_name"),
            "status": step_result.get("status"),
            "success": step_result.get("success"),
            "formatted": step_result.get("formatted"),
            "result_reference": step_result.get("result_reference"),
            "rows_preview": rows[:3],
            "error": step_result.get("error"),
        }

    @staticmethod
    def _parse_report_json(raw_text: str) -> AnalysisReport:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        payload = json.loads(cleaned)
        return AnalysisReport.model_validate(payload)

    def _build_fallback_report(
        self,
        original_question: str,
        analysis_goal: str,
        step_results: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> AnalysisReport:
        completed_results = [
            step_result for step_result in step_results
            if step_result.get("status") == "completed" and step_result.get("success")
        ]
        key_findings = [
            f"{step_result.get('step_name')}：{self._pick_step_brief(step_result)}"
            for step_result in completed_results
        ] or ["当前执行结果不足，暂时无法提炼关键发现。"]

        root_causes = [
            "目前先根据各步骤返回的结果做归纳，尚未引入额外业务规则。"
        ]
        if summary.get("failed_steps", 0) > 0:
            root_causes.append("部分步骤执行失败，当前归因只覆盖已成功返回的部分。")

        action_suggestions = [
            "优先复核关键产品线、区域或费用项的波动来源。",
            "继续补充失败步骤或缺失维度，再迭代报告结论。",
        ]
        report = AnalysisReport(
            title=original_question,
            executive_summary=(
                f"{analysis_goal}。"
                f"本次共完成 {summary.get('completed_steps', 0)} 个步骤，"
                f"失败 {summary.get('failed_steps', 0)} 个步骤，"
                f"跳过 {summary.get('skipped_steps', 0)} 个步骤。"
            ),
            key_findings=key_findings,
            root_causes=root_causes,
            trend_judgment=summary.get("summary_text", "当前结果可作为后续进一步分析的基础。"),
            action_suggestions=action_suggestions,
        )
        report.markdown = self._render_markdown(report)
        return report

    @staticmethod
    def _pick_step_brief(step_result: dict[str, Any]) -> str:
        formatted = (step_result.get("formatted") or "").strip()
        if formatted:
            return formatted.splitlines()[0][:120]
        rows = step_result.get("rows") or []
        if rows:
            return json.dumps(rows[0], ensure_ascii=False, default=ReportGenerator._json_default)
        error = step_result.get("error")
        if error:
            return f"执行失败：{error}"
        return "步骤已完成，但当前没有可展示的结果摘要。"

    @staticmethod
    def _render_markdown(report: AnalysisReport) -> str:
        sections = [
            f"# {report.title}",
            "",
            "## 执行摘要",
            report.executive_summary,
            "",
            "## 关键发现",
            *[f"- {finding}" for finding in report.key_findings],
            "",
            "## 归因分析",
            *[f"- {cause}" for cause in report.root_causes],
            "",
            "## 趋势判断",
            report.trend_judgment,
            "",
            "## 行动建议",
            *[f"- {suggestion}" for suggestion in report.action_suggestions],
        ]
        return "\n".join(sections)

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, Decimal):
            return str(value)
        return str(value)

    @staticmethod
    def _build_default_text_generator() -> ReportTextGenerator | None:
        if not LLM_CONFIG.get("api_key"):
            return None

        client = LLMClient()

        def _generate(system_msg: str, prompt: str) -> str:
            return client.generate_text(system_msg=system_msg, prompt=prompt)

        return _generate
