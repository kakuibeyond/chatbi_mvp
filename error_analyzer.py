"""
错误分析模块

负责分类 SQL 生成错误类型，提供错误根因分析与修复建议。
该模块在第7课引入，用于系统性地收集、分类和追踪 Text2SQL 系统的错误模式。
"""

import re
from typing import Optional


class ErrorAnalyzer:
    """SQL 生成错误分析器"""

    # 错误类型常量11
    FIELD_ERROR = "field_error"           # 字段选择错误
    JOIN_ERROR = "join_error"             # 关联路径错误
    TIME_ERROR = "time_error"             # 时间计算错误
    FILTER_ERROR = "filter_error"         # 过滤条件遗漏
    AGGREGATION_ERROR = "aggregation_error"  # 聚合逻辑错误
    SYNTAX_ERROR = "syntax_error"         # SQL 语法错误
    UNKNOWN = "unknown"                   # 未知错误

    def __init__(self):
        self.error_patterns = {
            self.FIELD_ERROR: [
                r"gross_amount",  # 错误使用含税金额
                r"standard_cost",  # 错误使用标准成本替代实际成本
            ],
            self.JOIN_ERROR: [
                r"exchange_rates",  # 关联汇率表的字段或条件错误
                r"JOIN.*ON.*=",     # 通用 Join 问题检测
            ],
            self.TIME_ERROR: [
                r"DATE_SUB",        # 时间函数使用问题
                r"CURDATE",         # 当前日期引用问题
            ],
            self.FILTER_ERROR: [
                r"order_status",    # 过滤条件相关
            ],
            self.AGGREGATION_ERROR: [
                r"GROUP BY",        # 聚合维度问题
                r"SUM\(|AVG\(|COUNT\(",  # 聚合函数问题
            ],
        }

    def categorize_error(
        self,
        sql: str,
        error_msg: Optional[str] = None,
        question: Optional[str] = None
    ) -> dict:
        """
        对 SQL 错误进行分类

        Args:
            sql: LLM 生成的 SQL 语句
            error_msg: 数据库执行返回的错误信息（如有）
            question: 用户的原始问题

        Returns:
            包含错误类型、根因分析和修复建议的字典
        """
        result = {
            "error_type": self.UNKNOWN,
            "root_cause": "",
            "suggestion": "",
            "sql": sql
        }

        # 如果有数据库执行错误，优先判断语法类错误
        if error_msg:
            if self._is_syntax_error(error_msg):
                result["error_type"] = self.SYNTAX_ERROR
                result["root_cause"] = f"SQL 语法错误：{error_msg}"
                result["suggestion"] = "检查 SQL 语句中的关键字拼写、括号匹配和引号使用，必要时在 Prompt 中增加语法约束。"
                return result

        # 基于 SQL 内容和问题语义进行启发式分类
        detected_types = self._detect_error_types(sql, question)

        if detected_types:
            primary_type = detected_types[0]
            result["error_type"] = primary_type
            result["root_cause"] = self._get_root_cause(primary_type, sql, question)
            result["suggestion"] = self._get_suggestion(primary_type)
        else:
            result["root_cause"] = "无法从 SQL 中直接识别错误模式，需人工审查"
            result["suggestion"] = "将该用例加入 Few-shot 示例或增强 Schema 描述中的相关字段说明"

        return result

    def _is_syntax_error(self, error_msg: str) -> bool:
        """判断是否为 SQL 语法错误"""
        syntax_keywords = [
            "syntax error", "parse error", "unexpected",
            "near", "expected", "invalid"
        ]
        return any(kw in error_msg.lower() for kw in syntax_keywords)

    def _detect_error_types(
        self,
        sql: str,
        question: Optional[str] = None
    ) -> list[str]:
        """基于启发式规则检测可能的错误类型"""
        detected = []
        sql_lower = sql.lower()

        # 字段选择错误检测
        if self._check_field_error(sql_lower, question):
            detected.append(self.FIELD_ERROR)

        # Join 错误检测
        if self._check_join_error(sql_lower, question):
            detected.append(self.JOIN_ERROR)

        # 时间错误检测
        if self._check_time_error(sql_lower, question):
            detected.append(self.TIME_ERROR)

        # 过滤遗漏检测
        if self._check_filter_error(sql_lower, question):
            detected.append(self.FILTER_ERROR)

        # 聚合错误检测
        if self._check_aggregation_error(sql_lower, question):
            detected.append(self.AGGREGATION_ERROR)

        return detected

    def _check_field_error(self, sql: str, question: Optional[str]) -> bool:
        """检测字段选择错误"""
        # 问题问收入但 SQL 用了 gross_amount
        if question and ("收入" in question or "销售额" in question):
            if "gross_amount" in sql and "net_amount" not in sql:
                return True
        # 问题问成本但 SQL 用了 standard_cost
        if question and "成本" in question:
            if "standard_cost" in sql and "material_cost" not in sql:
                return True
        return False

    def _check_join_error(self, sql: str, question: Optional[str]) -> bool:
        """检测 Join 关联错误"""
        # 涉及收入/销售额但未关联汇率表
        if question and ("收入" in question or "销售额" in question):
            if "exchange_rates" not in sql and "currency" in sql:
                return True
        # 涉及客户/产品维度但未关联对应维度表
        if question and ("客户" in question or "产品线" in question):
            if "dim_customers" not in sql and "dim_products" not in sql:
                return True
        return False

    def _check_time_error(self, sql: str, question: Optional[str]) -> bool:
        """检测时间计算错误"""
        if question and ("最近" in question or "上个" in question or "本月" in question):
            # 未使用日期函数处理动态时间
            if "DATE_SUB" not in sql and "CURDATE" not in sql:
                return True
        return False

    def _check_filter_error(self, sql: str, question: Optional[str]) -> bool:
        """检测过滤条件遗漏"""
        # 统计订单但未过滤状态
        if question and ("订单" in question or "收入" in question or "销售额" in question):
            if "order_status" not in sql:
                return True
        return False

    def _check_aggregation_error(self, sql: str, question: Optional[str]) -> bool:
        """检测聚合逻辑错误"""
        # 使用了聚合函数但没有 GROUP BY
        has_agg = bool(re.search(r'sum\(|avg\(|count\(|max\(|min\(', sql))
        has_groupby = "group by" in sql
        if has_agg and not has_groupby:
            # 检查 SELECT 中是否有非聚合字段（简单启发式）
            select_part = sql.split("from")[0] if "from" in sql else sql
            non_agg_fields = re.findall(r'select\s+(.*?)\s+from', sql, re.DOTALL)
            if non_agg_fields:
                fields = non_agg_fields[0].split(",")
                # 如果有多于一个字段且没有 GROUP BY，可能存在聚合错误
                if len(fields) > 1:
                    return True
        return False

    def _get_root_cause(
        self,
        error_type: str,
        sql: str,
        question: Optional[str]
    ) -> str:
        """根据错误类型生成根因描述"""
        causes = {
            self.FIELD_ERROR: "模型未能正确理解业务语义，选择了错误的字段",
            self.JOIN_ERROR: "模型遗漏了必要的表关联或使用了错误的关联条件",
            self.TIME_ERROR: "时间范围计算不符合业务预期，动态时间边界处理有误",
            self.FILTER_ERROR: "遗漏了关键的业务过滤条件，导致统计范围不正确",
            self.AGGREGATION_ERROR: "聚合维度与 SELECT 字段不匹配，或聚合逻辑有误",
            self.SYNTAX_ERROR: "生成的 SQL 存在语法问题，无法被数据库解析",
        }
        return causes.get(error_type, "未知原因")

    def _get_suggestion(self, error_type: str) -> str:
        """根据错误类型生成修复建议"""
        suggestions = {
            self.FIELD_ERROR: "在 RULES 中强化字段语义定义，增加字段选择错误的 Few-shot 负例",
            self.JOIN_ERROR: "在 ERROR_GUARDS 中补充 Join 路径检查清单，增加多表查询 Few-shot 示例",
            self.TIME_ERROR: "增加时间计算相关的 Few-shot 示例，明确动态时间边界的 SQL 写法",
            self.FILTER_ERROR: "在 RULES 中明确各类查询的强制过滤条件，增加过滤遗漏的负例",
            self.AGGREGATION_ERROR: "增加 GROUP BY 正确用法的示例，强调 SELECT 非聚合字段必须出现在 GROUP BY 中",
            self.SYNTAX_ERROR: "检查 Prompt 中是否有格式混乱导致模型输出不完整 SQL 的情况",
        }
        return suggestions.get(error_type, "人工审查该用例并补充到测试集")

    def analyze_batch(
        self,
        cases: list[dict]
    ) -> list[dict]:
        """
        批量分析多个错误案例

        Args:
            cases: 每个元素为 {"question": str, "sql": str, "error_msg": str} 的字典列表

        Returns:
            分析结果列表
        """
        results = []
        for case in cases:
            result = self.categorize_error(
                sql=case.get("sql", ""),
                error_msg=case.get("error_msg"),
                question=case.get("question")
            )
            result["question"] = case.get("question", "")
            results.append(result)
        return results

    def generate_report(self, results: list[dict]) -> str:
        """
        生成错误分析汇总报告

        Args:
            results: analyze_batch 返回的结果列表

        Returns:
            格式化的文本报告
        """
        from collections import Counter
        types = [r["error_type"] for r in results]
        type_counts = Counter(types)

        lines = [
            "=" * 50,
            "SQL 生成错误分析汇总报告",
            "=" * 50,
            f"总案例数：{len(results)}",
            "",
            "错误类型分布：",
        ]
        for err_type, count in type_counts.most_common():
            lines.append(f"  {err_type}: {count} 例")

        lines.extend(["", "详细案例："])
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] 问题：{r.get('question', 'N/A')}")
            lines.append(f"    SQL：{r['sql'][:80]}...")
            lines.append(f"    类型：{r['error_type']}")
            lines.append(f"    根因：{r['root_cause']}")
            lines.append(f"    建议：{r['suggestion']}")

        return "\n".join(lines)
