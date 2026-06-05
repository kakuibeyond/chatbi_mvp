"""
LLM 客户端模块

负责调用大模型 API 生成 SQL。
将 LLM 调用封装为独立模块，便于后续切换不同模型或增加重试逻辑。

第11课增强：新增 generate_sql_stream 流式方法，逐 chunk 产出 SQL 文本，
为 SSE 推送提供底层能力。保留原有 generate_sql 不动，确保向后兼容。
"""

import re
from typing import Generator
from openai import OpenAI
from config import LLM_CONFIG


class LLMClient:
    """LLM API 客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"]
        )
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]

    def generate_sql(self, system_msg: str, prompt: str) -> str:
        """
        调用 LLM 生成 SQL（同步，一次性返回）

        Args:
            system_msg: 系统角色消息
            prompt: 完整的用户 Prompt

        Returns:
            提取后的纯 SQL 字符串
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        raw_output = response.choices[0].message.content.strip()

        # 提取 SQL：去除 markdown 代码块标记
        sql = re.sub(r'```sql|```', '', raw_output).strip()

        return sql

    def generate_sql_stream(self, system_msg: str, prompt: str) -> Generator[str, None, None]:
        """
        调用 LLM 流式生成 SQL（逐 chunk 产出）

        每次_yield_一段增量文本，调用方可以实时推送给前端。
        流式输出结束后，调用方需要自行拼接完整 SQL 并做 markdown 代码块清理。

        Args:
            system_msg: 系统角色消息
            prompt: 完整的用户 Prompt

        Yields:
            增量文本片段（可能包含 markdown 代码块标记，需在完整拼接后统一清理）
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_content = delta.content if delta and delta.content is not None else None
            if delta_content is not None:
                yield delta_content

if __name__ == '__main__':
    llm = LLMClient()
    print(llm.generate_sql(system_msg='你是一个 SQL 生成器', prompt="查询学生数量"))