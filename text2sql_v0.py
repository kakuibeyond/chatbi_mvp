"""
第5课实战版本0：Zero-shot 基础版

仅提供角色定义和用户问题，不注入任何 Schema 信息。
展示无 Schema 信息时 LLM 生成 SQL 的准确性边界。

运行方式：
    uv run python text2sql_v0.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

import re
from openai import OpenAI


def generate_sql(user_question: str) -> str:
    """调用 LLM 生成 SQL（Zero-shot，无 Schema）"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    prompt = f"你是一个专业的 SQL 生成助手。\n\n问题：{user_question}\n\n请直接输出 SQL："

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000
    )

    raw_output = response.choices[0].message.content.strip()
    sql = re.sub(r'```sql|```', '', raw_output).strip()
    return sql


def run(question: str) -> None:
    """运行并输出生成的 SQL"""
    print(f"\n{'='*60}")
    print(f"问题：{question}")
    print(f"{'='*60}")
    sql = generate_sql(question)
    print(f"\n生成 SQL：\n{sql}")
    print("\n可将上述 SQL 复制到 MySQL 客户端执行验证。")


if __name__ == "__main__":
    questions = [
        "查询已完成订单的总数量",
        "按产品线统计总收入",
        "欧洲市场最近三个月的销售额是多少",
        "各产品线的毛利率是多少",
    ]
    for q in questions:
        run(q)
