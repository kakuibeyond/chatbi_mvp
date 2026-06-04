"""
主入口模块

整合所有模块，提供命令行交互界面。
这是 ChatBI Text2SQL 系统的统一入口，串联 query_parser → prompt_builder → llm_client → database → result_formatter 完整链路。
"""

import sys
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

    def run(self, user_question: str, use_few_shot: bool = True) -> dict:
        """
        运行完整链路

        Args:
            user_question: 用户自然语言问题
            use_few_shot: 是否使用 Few-shot

        Returns:
            包含 SQL、结果或错误信息的字典
        """
        # 1. 解析问题
        parsed = self.parser.parse(user_question)
        if not self.parser.validate(parsed):
            return {"success": False, "error": "输入问题为空"}

        # 2. 构造 Prompt
        system_msg, prompt = build_prompt(user_question, use_few_shot)

        # 3. 生成 SQL
        sql = self.llm.generate_sql(system_msg, prompt)

        # 4. 执行 SQL
        try:
            columns, results = self.db.execute(sql)
            formatted = self.formatter.format(columns, results)
            return {
                "success": True,
                "sql": sql,
                "columns": columns,
                "results": results,
                "formatted": formatted
            }
        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "error": str(e)
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
