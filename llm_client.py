"""
LLM 客户端模块

负责调用大模型 API 生成 SQL。
将 LLM 调用封装为独立模块，便于后续切换不同模型或增加重试逻辑。
"""

import re
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
        调用 LLM 生成 SQL

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
