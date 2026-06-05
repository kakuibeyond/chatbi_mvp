"""
主入口模块

整合所有模块，提供命令行交互界面。
这是 ChatBI Text2SQL 系统的统一入口，串联 query_parser → prompt_builder → llm_client → database → result_formatter 完整链路，
并整合第 7 课规则修复与第 9 课指标知识注入。
"""

import sys
from config import LLM_CONFIG
from indicator_knowledge import IndicatorKnowledge
from query_parser import QueryParser
from prompt_builder import build_prompt
from llm_client import LLMClient
from database import DatabaseClient
from result_formatter import ResultFormatter


class ChatBISystem:
    """ChatBI 系统主类"""

    def __init__(self):
        self.parser = QueryParser()
        self.llm = LLMClient()
        self.db = DatabaseClient()
        self.formatter = ResultFormatter()
        self.indicator_knowledge = IndicatorKnowledge()

    def run(
        self,
        user_question: str,
        use_few_shot: bool = True,
        use_rules: bool = True,
        use_guards: bool = True,
        use_indicator_knowledge: bool = True
    ) -> dict:
        """
        运行完整链路

        Args:
            user_question: 用户自然语言问题
            use_few_shot: 是否使用 Few-shot
            use_rules: 是否启用业务规则
            use_guards: 是否启用错误防护
            use_indicator_knowledge: 是否启用指标知识注入

        Returns:
            包含 SQL、结果或错误信息的字典
        """
        # 1. 解析问题
        parsed = self.parser.parse(user_question)
        if not self.parser.validate(parsed):
            return {
                "success": False,
                "error": "输入问题为空",
                "error_type": "validation"
            }

        detected_indicators = []
        indicator_block = ""
        if use_indicator_knowledge:
            detected_indicators = self.indicator_knowledge.detect_indicators(user_question)
            indicator_block = self.indicator_knowledge.build_knowledge_block(user_question)

        # 2. 构造 Prompt
        system_msg, prompt = build_prompt(
            user_question,
            use_few_shot=use_few_shot,
            use_rules=use_rules,
            use_guards=use_guards,
            indicator_knowledge=indicator_block
        )

        # 3. 生成 SQL
        try:
            sql = self.llm.generate_sql(system_msg, prompt)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "llm",
                "metadata": {
                    "detected_indicators": detected_indicators,
                    "model": LLM_CONFIG["model"],
                    "used_rules": use_rules,
                    "used_guards": use_guards,
                    "used_indicator_knowledge": use_indicator_knowledge,
                }
            }

        # 4. 执行 SQL
        try:
            columns, results = self.db.execute(sql)
            formatted = self.formatter.format(columns, results)
            return {
                "success": True,
                "sql": sql,
                "columns": columns,
                "results": results,
                "formatted": formatted,
                "metadata": {
                    "detected_indicators": detected_indicators,
                    "model": LLM_CONFIG["model"],
                    "used_few_shot": use_few_shot,
                    "used_rules": use_rules,
                    "used_guards": use_guards,
                    "used_indicator_knowledge": use_indicator_knowledge,
                    "row_count": len(results),
                }
            }
        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "error": str(e),
                "error_type": "database",
                "metadata": {
                    "detected_indicators": detected_indicators,
                    "model": LLM_CONFIG["model"],
                    "used_few_shot": use_few_shot,
                    "used_rules": use_rules,
                    "used_guards": use_guards,
                    "used_indicator_knowledge": use_indicator_knowledge,
                }
            }


def main():
    """命令行入口"""
    system = ChatBISystem()

    print("=" * 60)
    print("ChatBI Text2SQL 系统")
    print("=" * 60)

    # 命令行模式：直接传入问题
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = system.run(question)
        _print_result(question, result)
        return

    # 交互模式
    print("\n请输入问题（输入 exit / quit / q 退出）：")
    while True:
        try:
            question = input("\n> ")
            if question.strip().lower() in ["exit", "quit", "q"]:
                break
            result = system.run(question)
            _print_result(question, result)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"系统错误：{e}")

    print("\n感谢使用！")


def _print_result(question: str, result: dict):
    """打印执行结果"""
    print(f"\n问题：{question}")
    print(f"SQL：{result.get('sql', '')}")
    if result["success"]:
        print(f"\n{result['formatted']}")
    else:
        print(f"\n错误：{result['error']}")


if __name__ == "__main__":
    main()
