from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_planner import (
    PlanAndExecuteAgent,
    PlanGenerator,
    ResultSummarizer,
    StepExecutor,
    TempTableResultStore,
)
from report_generator import ReportGenerator


class FakeTempCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self._results = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.connection.executed.append((sql, params))
        sql_upper = " ".join(sql.upper().split())
        identifiers = re.findall(r"`([^`]+)`", sql)

        if sql_upper.startswith("DROP TEMPORARY TABLE IF EXISTS"):
            table_name = identifiers[0]
            self.connection.tables.pop(table_name, None)
            self.description = None
            self._results = []
            return

        if sql_upper.startswith("CREATE TEMPORARY TABLE"):
            table_name = identifiers[0]
            columns = identifiers[1:]
            self.connection.tables[table_name] = {"columns": columns, "rows": []}
            self.description = None
            self._results = []
            return

        if sql_upper.startswith("SELECT * FROM"):
            table_name = identifiers[0]
            table = self.connection.tables[table_name]
            columns = table["columns"]
            rows = table["rows"]
            self.description = [(column,) for column in columns]
            self._results = [tuple(row.get(column) for column in columns) for row in rows]
            return

        raise AssertionError(f"未处理的 SQL: {sql}")

    def executemany(self, sql, params_seq):
        params_list = list(params_seq)
        self.connection.executed.append((sql, params_list))
        identifiers = re.findall(r"`([^`]+)`", sql)
        table_name = identifiers[0]
        columns = identifiers[1:]
        table = self.connection.tables[table_name]
        table["rows"].extend(dict(zip(columns, params)) for params in params_list)

    def fetchall(self):
        return self._results


class FakeTempConnection:
    def __init__(self):
        self.tables = {}
        self.executed = []
        self.open = True

    def cursor(self):
        return FakeTempCursor(self)

    def commit(self):
        self.executed.append(("COMMIT", None))

    def close(self):
        self.open = False


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


def test_agent_returns_business_report_after_execution():
    decomposition = sample_decomposition()

    def fake_runner(question: str) -> dict:
        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": f"已执行：{question.splitlines()[0]}",
        }

    report_generator = ReportGenerator(
        text_generator=lambda _system_msg, _prompt: """
        {
          "title": "利润下降分析报告",
          "executive_summary": "利润下降主要来自收入回落。",
          "key_findings": ["最近一个月利润明显走低。"],
          "root_causes": ["收入下降快于成本下降。"],
          "trend_judgment": "短期仍需跟踪。",
          "action_suggestions": ["继续观察核心产品线订单恢复情况。"]
        }
        """
    )

    agent = PlanAndExecuteAgent(
        planner=PlanGenerator(),
        executor=StepExecutor(step_runner=fake_runner),
        summarizer=ResultSummarizer(),
        report_generator=report_generator,
    )

    result = agent.run(
        "最近三个月利润为什么下降？",
        decomposition_override=decomposition,
    )

    assert result["report"]["title"] == "利润下降分析报告"
    assert "## 关键发现" in result["report"]["markdown"]


def test_step_executor_retries_failed_step_and_records_result_reference():
    decomposition = sample_decomposition()
    plan = PlanGenerator().build_plan("最近三个月利润为什么下降？", decomposition)
    attempts = {"step_1": 0}

    def flaky_runner(question: str) -> dict:
        primary_instruction = question.splitlines()[0]
        if "查看最近三个月利润趋势" in primary_instruction:
            attempts["step_1"] += 1
            if attempts["step_1"] == 1:
                return {
                    "success": False,
                    "error": "数据库连接超时",
                    "formatted": "第一次执行失败",
                }

        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": "重试后执行成功",
        }

    executor = StepExecutor(
        step_runner=flaky_runner,
        max_retries=1,
        failure_policy="abort",
        storage_backend="memory",
    )
    results = executor.execute_plan(plan)

    assert attempts["step_1"] == 2
    assert results[0].success is True
    assert results[0].attempts == 2
    assert results[0].status == "completed"
    assert results[0].result_reference == "memory://step_1"


def test_step_executor_skips_downstream_steps_after_failed_dependency():
    decomposition = sample_decomposition()
    plan = PlanGenerator().build_plan("最近三个月利润为什么下降？", decomposition)

    def failing_runner(question: str) -> dict:
        primary_instruction = question.splitlines()[0]
        if "查看最近三个月利润趋势" in primary_instruction:
            return {
                "success": False,
                "error": "SQL 语法错误",
                "formatted": "首步失败",
            }

        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": "后续不应执行到这里",
        }

    executor = StepExecutor(
        step_runner=failing_runner,
        max_retries=0,
        failure_policy="skip",
    )
    results = executor.execute_plan(plan)

    assert results[0].status == "failed"
    assert results[1].status == "skipped"
    assert results[1].success is False
    assert "依赖步骤失败" in results[1].error
    assert results[2].status == "skipped"


def test_result_summarizer_counts_skipped_steps_separately():
    decomposition = sample_decomposition()
    plan = PlanGenerator().build_plan("最近三个月利润为什么下降？", decomposition)

    def failing_runner(question: str) -> dict:
        primary_instruction = question.splitlines()[0]
        if "查看最近三个月利润趋势" in primary_instruction:
            return {
                "success": False,
                "error": "SQL 语法错误",
                "formatted": "首步失败",
            }
        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": "后续不应执行到这里",
        }

    executor = StepExecutor(step_runner=failing_runner, failure_policy="skip")
    step_results = executor.execute_plan(plan)
    summary = ResultSummarizer().summarize(
        original_question="最近三个月利润为什么下降？",
        plan=plan,
        step_results=step_results,
    )

    assert summary.completed_steps == 0
    assert summary.failed_steps == 1
    assert summary.skipped_steps == 2


def test_temp_table_result_store_put_get_and_cleanup():
    fake_connection = FakeTempConnection()
    store = TempTableResultStore(connection_factory=lambda: fake_connection)

    reference = store.put(
        step_id="step_1",
        columns=["month", "profit"],
        rows=[{"month": "2026-05", "profit": 920000}],
    )
    loaded_rows = store.get(reference)

    assert reference == "temp_table://tmp_agent_step_1"
    assert loaded_rows == [{"month": "2026-05", "profit": 920000}]

    store.cleanup()

    assert fake_connection.open is False
    assert fake_connection.tables == {}


def test_step_executor_can_load_rows_from_temp_table_reference():
    decomposition = sample_decomposition()
    plan = PlanGenerator().build_plan("最近三个月利润为什么下降？", decomposition)
    fake_connection = FakeTempConnection()
    captured_questions: list[str] = []

    def runner(question: str) -> dict:
        captured_questions.append(question)
        primary_instruction = question.splitlines()[0]
        if "查看最近三个月利润趋势" in primary_instruction:
            return {
                "success": True,
                "sql": "SELECT month, profit FROM profit_trend",
                "columns": ["month", "profit"],
                "rows": [{"month": "2026-05", "profit": 920000}],
                "formatted": "",
            }

        return {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["value"],
            "rows": [{"value": 1}],
            "formatted": "后续执行成功",
        }

    executor = StepExecutor(
        step_runner=runner,
        storage_backend="temp_table",
        storage_connection_factory=lambda: fake_connection,
    )
    results = executor.execute_plan(plan, max_steps=2)
    loaded_rows = executor.get_intermediate_result(results[0].result_reference)

    assert results[0].result_reference == "temp_table://tmp_agent_step_1"
    assert loaded_rows == [{"month": "2026-05", "profit": 920000}]
    assert '{"month": "2026-05", "profit": 920000}' in captured_questions[1]
