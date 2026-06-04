"""
结果格式化模块

负责将数据库执行结果格式化为可读输出。
当前版本输出文本表格，后续课程中可扩展为 JSON、Markdown 表格等格式。
"""

from typing import List, Tuple


class ResultFormatter:
    """结果格式化器"""

    def format(self, columns: List[str], results: List[tuple]) -> str:
        """
        将查询结果格式化为字符串表格

        Args:
            columns: 列名列表
            results: 结果行列表

        Returns:
            格式化后的字符串
        """
        if not results:
            return "查询结果为空"

        # 计算每列最大宽度
        col_widths = []
        for i, col in enumerate(columns):
            max_data_width = max(len(str(row[i])) for row in results)
            col_widths.append(max(len(col), max_data_width) + 2)

        # 构建表头
        header = "|".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
        separator = "+".join("-" * w for w in col_widths)

        # 构建数据行
        rows = []
        for row in results:
            row_str = "|".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
            rows.append(row_str)

        return (
            f"{separator}\n{header}\n{separator}\n"
            + "\n".join(rows)
            + f"\n{separator}"
        )

    def format_error(self, error_msg: str) -> str:
        """
        格式化错误信息

        Args:
            error_msg: 错误消息

        Returns:
            格式化后的错误字符串
        """
        return f"执行出错：{error_msg}"
