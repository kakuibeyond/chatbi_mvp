from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import LLMClient
from prompt_builder import build_prompt


def test_build_prompt_keeps_curdate_rule_for_recent_months():
    _system_msg, prompt = build_prompt("最近三个月利润为什么下降？", use_rules=True)

    assert "DATE_SUB(CURDATE(), INTERVAL N MONTH)" in prompt


def test_llm_client_allows_longer_sql_output():
    client = LLMClient()

    assert client.max_tokens >= 4000
