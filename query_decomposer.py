"""
Query 拆解器模块

将复杂分析问题拆解为结构化子任务列表，为后续 Planner 提供稳定输入。
"""

import json
from pathlib import Path
import re
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from llm_client import LLMClient
from prompt_builder import SCHEMA


class DecomposedTask(BaseModel):
    """单个子任务定义。"""

    task_id: str
    task_name: str
    task_type: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class DecompositionPlan(BaseModel):
    """复杂查询拆解结果。"""

    question_type: str
    analysis_goal: str
    subtasks: list[DecomposedTask]


AVAILABLE_DIMENSIONS = [
    "月份",
    "客户",
    "客户类型",
    "行业",
    "国家",
    "区域",
    "产品",
    "产品线",
    "品类", # 品种类型 category
    "技术路线",
    "部门",
    "费用项目",
    "原因类型",
]


def _load_indicator_catalog() -> list[str]:
    config_path = Path(__file__).with_name("indicators_full.json")
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    names: list[str] = []
    for indicator in data.get("indicators", []):
        names.append(indicator["name"])
        names.extend(indicator.get("aliases", []))
    return names


def _max_tasks_for_question_type(question_type: str) -> int:
    normalized = question_type.lower()
    if "trend" in normalized or "趋势" in question_type:
        return 6
    if "diagnosis" in normalized or "原因" in question_type or "下降" in question_type:
        return 6
    if "dimension" in normalized or "维度" in question_type:
        return 5
    return 8


def build_decomposition_prompt(user_question: str) -> tuple[str, str]:
    """构造 Query 拆解 Prompt。"""
    indicator_catalog = "、".join(_load_indicator_catalog())
    dimension_catalog = "、".join(AVAILABLE_DIMENSIONS)
    system_msg = (
        "你是企业级 ChatBI 系统中的任务拆解器。"
        "请把复杂分析问题拆成可执行的子任务列表，"
        "输出必须是 JSON，不要输出额外解释。"
    )
    prompt = f"""
【数据库 Schema】
{SCHEMA}

【可用分析维度】
{dimension_catalog}

【可用指标】
{indicator_catalog}

请将下面的复杂分析问题拆解为结构化子任务，并严格输出 JSON：

用户问题：{user_question}

输出要求：
1. 顶层字段包含 question_type、analysis_goal、subtasks
2. subtasks 是有序数组，每个任务必须包含：
   - task_id
   - task_name
   - task_type
   - description
   - depends_on
   - dimensions
   - metrics
3. task_id 使用 task_1、task_2 这类格式
4. depends_on 只能引用前面已经出现的 task_id
5. 每个子任务都应尽量对应一条简单 SQL 或一个单独分析动作，不要把多个分析都塞进同一步
6. 仅返回 JSON 对象，不要使用 Markdown
7. 维度只能从【可用分析维度】中选择；如用户问题提到未建模维度，应回退到最接近的已建模维度
8. 指标优先复用【可用指标】中的名称或别名
9. 如果一个任务同时要求多个维度或多个驱动因素，请拆成多个更简单的子任务

【常见分析类型拆解策略】
1. 趋势分析：优先拆成“主指标趋势 + 关键驱动项趋势 + 异常月份定位 + 原因归纳”，通常不超过 6 个子任务
2. 原因诊断：优先拆成“结果趋势 + 构成拆解 + 关键维度定位 + 必要专项分析 + 结论汇总”，通常不超过 6 个子任务
3. 维度对比：优先拆成“核心维度排名/贡献 + 必要补充维度 + 结论汇总”，通常不超过 5 个子任务
""".strip()
    return system_msg, prompt


class QueryDecomposer:
    """复杂问题拆解器。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        response_generator: Callable[[str, str], str] | None = None,
    ):
        self.llm = llm_client or LLMClient()
        self.response_generator = response_generator or self._generate_response

    def decompose(self, user_question: str) -> dict:
        """将复杂问题拆解为结构化子任务。"""
        question = user_question.strip()
        if not question:
            raise ValueError("输入问题不能为空")

        system_msg, prompt = build_decomposition_prompt(question)
        last_error: ValueError | None = None

        for attempt in range(2):
            raw_response = self.response_generator(system_msg, prompt)
            print(f"LLM 原始输出: \n{raw_response}")
            plan = self._parse_plan(raw_response)
            print(f"解析后的子任务: \n{plan}")

            try:
                self._validate_dependencies(plan)
                self._validate_dimensions(plan)
                self._validate_plan_complexity(plan)
                return plan.model_dump()
            except ValueError as exc:
                last_error = exc
                if attempt == 1:
                    raise
                prompt = self._build_retry_prompt(prompt, str(exc))

        assert last_error is not None
        raise last_error

    def _generate_response(self, system_msg: str, prompt: str) -> str:
        """默认通过 LLM 生成 JSON 结果。"""
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM 没有返回可用的拆解结果")
        return content

    def _parse_plan(self, raw_response: str) -> DecompositionPlan:
        """解析 LLM 返回结果，兼容 Markdown 代码块。"""
        json_text = self._extract_json(raw_response)
        try:
            return DecompositionPlan.model_validate_json(json_text)
        except ValidationError as exc:
            raise ValueError(f"拆解结果结构不合法: {exc}") from exc

    @staticmethod
    def _extract_json(raw_response: str) -> str:
        """从模型输出中提取 JSON 文本。"""
        text = raw_response.strip()
        fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as exc:
            raise ValueError("LLM 返回的内容不是合法 JSON") from exc

    @staticmethod
    def _validate_dependencies(plan: DecompositionPlan) -> None:
        """校验依赖任务是否存在且顺序合法。"""
        seen_ids: set[str] = set()
        for task in plan.subtasks:
            for dep_id in task.depends_on:
                if dep_id not in seen_ids:
                    raise ValueError(f"任务 {task.task_id} 依赖了不存在的任务: {dep_id}")
            seen_ids.add(task.task_id)

    def _normalize_plan(
        self,
        user_question: str,
        plan: DecompositionPlan,
    ) -> DecompositionPlan:
        return plan

    @staticmethod
    def _validate_dimensions(plan: DecompositionPlan) -> None:
        unsupported = sorted(
            {
                dimension
                for task in plan.subtasks
                for dimension in task.dimensions
                if dimension not in AVAILABLE_DIMENSIONS
            }
        )
        if unsupported:
            raise ValueError(f"不支持的分析维度: {', '.join(unsupported)}")

    @staticmethod
    def _validate_plan_complexity(plan: DecompositionPlan) -> None:
        max_tasks = _max_tasks_for_question_type(plan.question_type)
        if len(plan.subtasks) > max_tasks:
            raise ValueError(
                f"子任务过多：当前 question_type={plan.question_type} 最多允许 {max_tasks} 个子任务，"
                f"实际返回 {len(plan.subtasks)} 个。请合并重复任务并保持每步简单。"
            )

    @staticmethod
    def _build_retry_prompt(prompt: str, error_message: str) -> str:
        return (
            f"{prompt}\n\n"
            "上一次拆解结果不合法，存在以下问题：\n"
            f"{error_message}\n"
            "请重新拆解，并严格修正上述问题后只返回 JSON。"
        )


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or "最近三个月利润为什么下降？"
    decomposer = QueryDecomposer()
    result = decomposer.decompose(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
