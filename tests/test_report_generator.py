import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_generator import ReportGenerator


def sample_step_results() -> list[dict]:
    return [
        {
            "step_id": "step_1",
            "step_name": "查看最近三个月利润趋势",
            "success": True,
            "status": "completed",
            "formatted": "2026-05 的利润最低，为 920000 元。",
            "rows": [{"month": "2026-05", "profit": 920000}],
            "result_reference": "memory://step_1",
        },
        {
            "step_id": "step_2",
            "step_name": "拆解收入与成本变化",
            "success": True,
            "status": "completed",
            "formatted": "动力电池-乘用车收入下降 8%，材料成本上升 5%。",
            "rows": [{"product_line": "动力电池-乘用车", "revenue_yoy": -0.08, "material_cost_yoy": 0.05}],
            "result_reference": "memory://step_2",
        },
    ]


def test_report_generator_parses_structured_llm_output():
    def fake_text_generator(system_msg: str, prompt: str) -> str:
        assert "关键发现" in prompt
        return json.dumps(
            {
                "title": "最近三个月利润下降分析报告",
                "executive_summary": "利润下滑主要来自动力电池-乘用车业务收入回落和材料成本上升。",
                "key_findings": [
                    "2026-05 的利润最低。",
                    "动力电池-乘用车业务收入下降 8%。",
                ],
                "root_causes": [
                    "主要产品线收入回落。",
                    "材料成本继续上升。",
                ],
                "trend_judgment": "短期利润压力仍在，需继续跟踪 6 月订单修复情况。",
                "action_suggestions": [
                    "优先复盘动力电池-乘用车订单流失原因。",
                    "单独跟踪材料采购价格与毛利变化。",
                ],
            },
            ensure_ascii=False,
        )

    generator = ReportGenerator(text_generator=fake_text_generator)
    report = generator.generate(
        original_question="最近三个月利润为什么下降？",
        analysis_goal="定位最近三个月利润下降的主要驱动因素",
        step_results=sample_step_results(),
        summary={
            "completed_steps": 2,
            "failed_steps": 0,
            "skipped_steps": 0,
            "summary_text": "当前结果已经可以支撑后续的中间结果管理与最终报告生成。",
        },
    )

    assert report.title == "最近三个月利润下降分析报告"
    assert report.key_findings[0] == "2026-05 的利润最低。"
    assert "## 关键发现" in report.markdown
    assert "## 行动建议" in report.markdown


def test_report_generator_falls_back_to_template_when_llm_output_is_invalid():
    generator = ReportGenerator(text_generator=lambda _system_msg, _prompt: "not-json")
    report = generator.generate(
        original_question="最近三个月利润为什么下降？",
        analysis_goal="定位最近三个月利润下降的主要驱动因素",
        step_results=sample_step_results(),
        summary={
            "completed_steps": 2,
            "failed_steps": 0,
            "skipped_steps": 0,
            "summary_text": "当前结果已经可以支撑后续的中间结果管理与最终报告生成。",
        },
    )

    assert report.title == "最近三个月利润为什么下降？"
    assert report.key_findings[0].startswith("查看最近三个月利润趋势：")
    assert "动力电池-乘用车收入下降 8%" in report.markdown
