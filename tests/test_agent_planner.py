from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_planner import (
    PlanAndExecuteAgent,
    PlanGenerator,
    ResultSummarizer,
    StepExecutor,
)


def sample_decomposition() -> dict:
    return {
        "question_type": "profit_decline_analysis",
        "analysis_goal": "定位最近三个月利润下降的主要驱动因素",
        "subtasks": [
            {
                "task_id": "task_1",
                "task_name": "查看最近三个月利润趋势",
                "task_type": "trend_analysis",
                "description": "先找出利润下降最明显的月份",
                "depends_on": [],
                "dimensions": ["月份"],
                "metrics": ["利润"],
            },
            {
                "task_id": "task_2",
                "task_name": "拆解收入与成本变化",
                "task_type": "metric_decomposition",
                "description": "围绕利润 = 收入 - 成本，确认是哪一端拖累了利润",
                "depends_on": ["task_1"],
                "dimensions": ["月份"],
                "metrics": ["收入", "成本", "利润"],
            },
            {
                "task_id": "task_3",
                "task_name": "定位利润下滑最严重的区域",
                "task_type": "dimension_drilldown",
                "description": "结合前两步结果，观察哪个区域的利润恶化更明显",
                "depends_on": ["task_2"],
                "dimensions": ["区域"],
                "metrics": ["利润", "收入", "成本"],
            },
        ],
    }


def test_plan_generator_builds_ordered_steps():
    decomposition = sample_decomposition()
    planner = PlanGenerator()

    plan = planner.build_plan(
        original_question="最近三个月利润为什么下降？",
        decomposition=decomposition,
    )

    assert plan.analysis_goal == "定位最近三个月利润下降的主要驱动因素"
    assert [step.step_id for step in plan.steps] == ["step_1", "step_2", "step_3"]
    assert plan.steps[1].depends_on == ["step_1"]
    assert "关注指标：收入、成本、利润" in plan.steps[1].question


def test_step_executor_passes_dependency_context_to_later_steps():
    decomposition = sample_decomposition()
    planner = PlanGenerator()
    plan = planner.build_plan("最近三个月利润为什么下降？", decomposition)
    captured_questions: list[str] = []

    def fake_runner(question: str) -> dict:
        captured_questions.append(question)
        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": "模拟执行成功",
        }

    executor = StepExecutor(step_runner=fake_runner)
    results = executor.execute_plan(plan)

    assert len(results) == 3
    assert "前置步骤关键结果如下" not in captured_questions[0]
    assert "查看最近三个月利润趋势" in captured_questions[1]
    assert "拆解收入与成本变化" in captured_questions[2]


def test_agent_returns_summary_after_execution():
    decomposition = sample_decomposition()

    def fake_runner(question: str) -> dict:
        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": f"已执行：{question.splitlines()[0]}",
        }

    agent = PlanAndExecuteAgent(
        planner=PlanGenerator(),
        executor=StepExecutor(step_runner=fake_runner),
        summarizer=ResultSummarizer(),
    )

    result = agent.run(
        "最近三个月利润为什么下降？",
        decomposition_override=decomposition,
    )

    assert result["summary"]["completed_steps"] == 3
    assert result["summary"]["failed_steps"] == 0
    assert len(result["summary"]["key_findings"]) == 3
    assert result["plan"]["steps"][2]["depends_on"] == ["step_2"]
