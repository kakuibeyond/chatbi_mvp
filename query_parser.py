"""
查询解析模块

负责对用户输入进行基础解析和校验。
当前版本仅做简单校验，后续课程中可扩展为意图识别、实体提取等复杂功能。
"""


class QueryParser:
    """用户查询解析器"""

    def parse(self, user_input: str) -> dict:
        """
        解析用户输入

        Args:
            user_input: 原始用户输入

        Returns:
            解析结果字典
        """
        return {
            "original_question": user_input.strip(),
            "is_valid": len(user_input.strip()) > 0
        }

    def validate(self, parsed_query: dict) -> bool:
        """
        校验解析结果是否有效

        Args:
            parsed_query: parse 方法的输出

        Returns:
            是否通过校验
        """
        return parsed_query.get("is_valid", False)
