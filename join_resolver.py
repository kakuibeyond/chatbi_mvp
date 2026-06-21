"""
多表 Join 路径自动推理模块

第 17 课：在表召回和字段匹配之后，自动推导多表之间的 Join 路径和 Join 条件。

本模块包含两个独立职责：
1. 锚表选择（select_anchor）：基于用户问题的意图，判断哪张表应该作为 SQL 的 FROM 子句主表。
2. Join 路径推导（resolve_joins）：已知锚表后，使用 BFS 图算法找到连接所有目标表的最短路径。

本课不使用 Embedding，是纯图算法 + 轻量意图识别。
"""

from collections import deque


# ==================== 表关联图配置 ====================
# 手动维护表间关系，原因：
# 1. 企业数据库常不建外键约束
# 2. 需要区分 JOIN / LEFT JOIN
# 3. 支持复合键
TABLE_RELATIONSHIPS = {
    "sales_orders": [
        {
            "target": "dim_customers",
            "fk_col": "customer_id",
            "pk_col": "customer_id",
            "join_type": "JOIN",
        },
        {
            "target": "dim_products",
            "fk_col": "product_id",
            "pk_col": "product_id",
            "join_type": "JOIN",
        },
        {
            "target": "exchange_rates",
            "fk_col": "order_date, currency",
            "pk_col": "rate_date, currency",
            "join_type": "LEFT JOIN",
        },
    ],
    # 从维度表出发找事实表时，应使用 LEFT JOIN：不是所有维度值都有对应的事实记录
    "dim_customers": [
        {
            "target": "sales_orders",
            "fk_col": "customer_id",
            "pk_col": "customer_id",
            "join_type": "LEFT JOIN",
        },
    ],
    "dim_products": [
        {
            "target": "sales_orders",
            "fk_col": "product_id",
            "pk_col": "product_id",
            "join_type": "LEFT JOIN",
        },
    ],
    "exchange_rates": [
        {
            "target": "sales_orders",
            "fk_col": "rate_date, currency",
            "pk_col": "order_date, currency",
            "join_type": "LEFT JOIN",
        },
    ],
    # 费用表与订单表在业务上独立，不建立直接关联
    "finance_expenses": [],
}


# ==================== 表类型与业务关键词 ====================
TABLE_TYPES = {
    "sales_orders": "fact",
    "finance_expenses": "fact",
    "dim_customers": "dimension",
    "dim_products": "dimension",
    "exchange_rates": "reference",
}

# 用于锚表选择的关键词匹配
TABLE_KEYWORDS = {
    "sales_orders": ["收入", "销售额", "订单", "销售", "数量", "金额", "毛利", "利润", "net_amount", "gross_amount"],
    "finance_expenses": ["费用", "研发", "销售费用", "管理费用", "财务费用", "期间费用", "expense"],
    "dim_customers": ["客户", "客户类型", "OEM", "储能集成商", "电网集团", "customer"],
    "dim_products": ["产品", "产品线", "型号", "SKU", "product"],
    "exchange_rates": ["汇率", "币种", "人民币", "美元", "欧元", "rate"],
}

# 查询意图信号词
METRIC_SIGNALS = ["统计", "多少", "总计", "平均", "占比", "总和", "额", "量"]
ENTITY_SIGNALS = ["列出", "哪些", "所有", "没有", "明细", "每个"]
STRONG_METRIC_WORDS = ["收入", "销售额", "费用", "毛利", "利润", "金额", "数量"]


# ==================== 锚表选择 ====================
def _classify_intent(query: str) -> str:
    """判断查询是指标型、实体型还是模糊型"""
    metric_count = sum(1 for s in METRIC_SIGNALS if s in query)
    entity_count = sum(1 for s in ENTITY_SIGNALS if s in query)

    if any(w in query for w in STRONG_METRIC_WORDS):
        metric_count += 2

    if metric_count > entity_count:
        return "metric"
    if entity_count > metric_count:
        return "entity"
    return "ambiguous"


def _score_tables_by_keywords(query: str, candidate_tables: list[str]) -> dict[str, int]:
    """根据关键词重叠度为每张候选表打分"""
    scores = {}
    for table in candidate_tables:
        keywords = TABLE_KEYWORDS.get(table, [])
        scores[table] = sum(1 for kw in keywords if kw in query)
    return scores


def _bfs_shortest_path(start: str, end: str, relationships: dict) -> list[str] | None:
    """BFS 寻找从 start 到 end 的最短路径"""
    if start == end:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        for edge in relationships.get(current, []):
            next_table = edge["target"]
            if next_table in visited:
                continue
            new_path = path + [next_table]
            if next_table == end:
                return new_path
            visited.add(next_table)
            queue.append((next_table, new_path))

    return None


def _is_connected(anchor: str, target_tables: list[str], relationships: dict) -> bool:
    """检查锚表是否能到达所有目标表"""
    for target in target_tables:
        if target != anchor and _bfs_shortest_path(anchor, target, relationships) is None:
            return False
    return True


def _most_central_table(candidate_tables: list[str], relationships: dict) -> str:
    """选择子图中到达其他节点总距离最小的表作为兜底锚表"""
    total_distances = {}
    for table in candidate_tables:
        total = 0
        for target in candidate_tables:
            if table == target:
                continue
            path = _bfs_shortest_path(table, target, relationships)
            total += len(path) - 1 if path else 999
        total_distances[table] = total
    return min(candidate_tables, key=lambda t: total_distances[t])


def select_anchor(query: str, candidate_tables: list[str], relationships: dict = None) -> tuple[str, str]:
    """
    根据用户问题选择锚表。

    锚表 = SQL 中 FROM 子句的第一张表。它应该对应用户问题的语义主体：
    - 指标型问题（问收入、费用、毛利等）→ 锚表是承载该指标的事实表
    - 实体型问题（问客户、产品、汇率等）→ 锚表是承载该实体的维度/参考表

    Args:
        query: 用户自然语言问题
        candidate_tables: 候选表名列表（通常来自 table_retriever）
        relationships: 表关联图，默认使用 TABLE_RELATIONSHIPS

    Returns:
        (anchor_table, reason)
    """
    if relationships is None:
        relationships = TABLE_RELATIONSHIPS

    if not candidate_tables:
        raise ValueError("候选表集合不能为空")

    if len(candidate_tables) == 1:
        return candidate_tables[0], "仅有一张候选表"

    intent = _classify_intent(query)
    scores = _score_tables_by_keywords(query, candidate_tables)
    fact_tables = [t for t in candidate_tables if TABLE_TYPES.get(t) == "fact"]
    dim_tables = [t for t in candidate_tables if TABLE_TYPES.get(t) != "fact"]

    anchor = None
    reason = ""

    if intent == "metric" and fact_tables:
        anchor = max(fact_tables, key=lambda t: scores.get(t, 0))
        reason = f"问题询问指标，在事实表中选择最相关的 '{anchor}' 作为锚表"
    elif intent == "entity" and dim_tables:
        anchor = max(dim_tables, key=lambda t: scores.get(t, 0))
        reason = f"问题询问实体，在维度/参考表中选择最相关的 '{anchor}' 作为锚表"
    else:
        anchor = _most_central_table(candidate_tables, relationships)
        reason = f"查询意图较模糊，选择子图中心性最高的 '{anchor}' 作为锚表"

    # 如果选中的锚表无法连通所有目标，按关键词分数从高到低尝试其他候选
    if not _is_connected(anchor, candidate_tables, relationships):
        for candidate in sorted(candidate_tables, key=lambda t: scores.get(t, 0), reverse=True):
            if _is_connected(candidate, candidate_tables, relationships):
                reason += f"；原锚表无法连通所有目标表，切换为 '{candidate}'"
                anchor = candidate
                break

    return anchor, reason


# ==================== Join 路径推导 ====================
def _get_edge_info(from_table: str, to_table: str, relationships: dict) -> dict:
    """获取两个表之间的边信息"""
    for edge in relationships.get(from_table, []):
        if edge["target"] == to_table:
            return edge
    raise ValueError(f"未找到从 {from_table} 到 {to_table} 的关联关系")


def _build_on_clause(from_table: str, to_table: str, edge: dict) -> str:
    """根据边信息生成 ON 子句"""
    fk_cols = [c.strip() for c in edge["fk_col"].split(",")]
    pk_cols = [c.strip() for c in edge["pk_col"].split(",")]
    conditions = []
    for fk, pk in zip(fk_cols, pk_cols):
        conditions.append(f"{from_table}.{fk} = {to_table}.{pk}")
    return " AND ".join(conditions)


def resolve_joins(anchor_table: str, target_tables: list[str], relationships: dict = None) -> dict:
    """
    从锚表出发，推导连接所有目标表的 Join 路径。

    Args:
        anchor_table: SQL FROM 子句的主表
        target_tables: 需要关联的所有表名
        relationships: 表关联图，默认使用 TABLE_RELATIONSHIPS

    Returns:
        {
            "anchor": str,
            "joins": [dict],
            "unreachable": [str],
            "sql_fragment": str,
        }
    """
    if relationships is None:
        relationships = TABLE_RELATIONSHIPS

    all_targets = set(target_tables)
    all_targets.add(anchor_table)

    visited_edges = set()
    joins = []
    unreachable = []

    for target in all_targets:
        if target == anchor_table:
            continue

        path = _bfs_shortest_path(anchor_table, target, relationships)
        if path is None:
            unreachable.append(target)
            continue

        for i in range(len(path) - 1):
            from_t, to_t = path[i], path[i + 1]
            edge_key = (from_t, to_t)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            edge = _get_edge_info(from_t, to_t, relationships)
            joins.append({
                "from_table": from_t,
                "to_table": to_t,
                "join_type": edge["join_type"],
                "on_clause": _build_on_clause(from_t, to_t, edge),
            })

    sql_fragment = _build_sql_fragment(anchor_table, joins)

    return {
        "anchor": anchor_table,
        "joins": joins,
        "unreachable": unreachable,
        "sql_fragment": sql_fragment,
    }


def _build_sql_fragment(anchor_table: str, joins: list[dict]) -> str:
    """生成 SQL FROM/JOIN 片段"""
    # 语义上先写 INNER JOIN，再写 LEFT JOIN，避免后续表把前面的 LEFT JOIN 结果过滤掉
    sorted_joins = sorted(joins, key=lambda j: 0 if j["join_type"] == "JOIN" else 1)
    lines = [anchor_table]
    for j in sorted_joins:
        lines.append(f"{j['join_type']} {j['to_table']} ON {j['on_clause']}")
    return "\n".join(lines)


# ==================== 演示 ====================
if __name__ == "__main__":
    test_cases = [
        {
            "query": "按客户类型统计收入",
            "tables": ["sales_orders", "dim_customers"],
        },
        {
            "query": "列出所有客户",
            "tables": ["dim_customers"],
        },
        {
            "query": "哪些客户没有下过订单",
            "tables": ["dim_customers", "sales_orders"],
        },
        {
            "query": "各产品线的毛利率",
            "tables": ["sales_orders", "dim_products"],
        },
        {
            "query": "上个月的研发费用",
            "tables": ["finance_expenses"],
        },
        {
            "query": "按客户类型统计各产品线的收入，需要换算成人民币",
            "tables": ["sales_orders", "dim_customers", "dim_products", "exchange_rates"],
        },
        {
            "query": "销售收入和期间费用对比",
            "tables": ["sales_orders", "finance_expenses"],
        },
    ]

    print("=" * 70)
    print("多表 Join 路径自动推理演示")
    print("=" * 70)

    for case in test_cases:
        query = case["query"]
        tables = case["tables"]

        print(f"\n问题：{query}")
        print(f"候选表：{tables}")

        anchor, reason = select_anchor(query, tables)
        print(f"锚表：{anchor}（{reason}）")

        result = resolve_joins(anchor, tables)

        if result["unreachable"]:
            print(f"⚠ 无法连通的表：{result['unreachable']}")
            print("  这些表需要独立查询，不能通过 Join 关联")
        else:
            print("生成 SQL 片段：")
            for line in result["sql_fragment"].split("\n"):
                print(f"  {line}")
