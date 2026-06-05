"""
向量检索演示 Demo

第 14 课教学代码：手写最简向量检索系统。
演示 Embedding 生成 → 向量存储 → 余弦相似度检索 的完整闭环。
本文件为教学演示用途，不纳入主系统模块。
"""

import numpy as np
from llm_client import LLMClient


# ==================== 1. 表描述数据 ====================
# 为每张表构建一段"自然语言描述"，这段文字将被转为向量
TABLE_DESCRIPTIONS = {
    "dim_customers": (
        "客户维度表：存储客户基本信息，包括客户名称、客户类型"
        "（OEM整车厂、储能集成商、电网集团等）、所属行业（交通、能源、工业）、"
        "所在国家和销售大区（欧洲、北美、亚太等）。"
        "用于按客户维度分析收入、利润和订单分布。"
    ),
    "dim_products": (
        "产品维度表：存储产品主数据，包括产品名称、产品线"
        "（动力电池-乘用车、储能系统-电网级等）、技术路线（三元锂、磷酸铁锂等）、"
        "以及产品成本信息（标准成本、材料成本、人工成本）。"
        "用于按产品维度分析收入、毛利、成本结构。"
    ),
    "sales_orders": (
        "销售订单表：记录每笔销售订单的详细信息，包括订单日期、订单状态、"
        "数量、单价、折扣、含税总额（gross_amount）、不含税收入（net_amount）、币种。"
        "通过 customer_id 和 product_id 关联客户表和产品表。"
        "是收入分析、订单统计的核心事实表。"
    ),
    "exchange_rates": (
        "汇率表：按日期和币种记录兑人民币汇率（rate_to_cny）。"
        "当订单涉及多币种时，需关联此表将金额统一折算为人民币。"
        "用于多币种收入汇总和跨区域财务对比。"
    ),
    "finance_expenses": (
        "费用表：记录企业各部门的期间费用明细，包括研发费用、销售费用、"
        "管理费用、财务费用，以及销售费用的子项（市场费用、物流费用、质保费用）。"
        "用于费用分析、利润计算（利润 = 毛利 - 期间费用）。"
    ),
}


# ==================== 2. 余弦相似度计算 ====================
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度。

    余弦相似度 = (A·B) / (|A| × |B|)
    - 值域 [-1, 1]，越接近 1 表示越相似
    - 语义相近的文本，其 Embedding 向量的余弦相似度通常 > 0.7
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


# ==================== 3. 构建向量索引 ====================
def build_table_index(client: LLMClient) -> dict[str, list[float]]:
    """
    将所有表的描述文本转为 Embedding 向量，构建内存索引。

    Returns:
        {表名: embedding向量} 的字典
    """
    print("正在构建表描述向量索引...")
    index = {}
    for table_name, description in TABLE_DESCRIPTIONS.items():
        embedding = client.get_embedding(description)
        index[table_name] = embedding
        print(f"  ✓ {table_name} → 向量维度 {len(embedding)}")
    print(f"索引构建完成，共 {len(index)} 张表\n")
    return index


# ==================== 4. 向量检索 ====================
def search_tables(
    query: str,
    index: dict[str, list[float]],
    client: LLMClient,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    根据用户问题检索最相关的表。

    Args:
        query: 用户的自然语言问题
        index: 表名→向量 的索引字典
        client: LLM 客户端（用于生成 query embedding）
        top_k: 返回相似度最高的前 K 张表

    Returns:
        [(表名, 相似度分数)] 列表，按分数降序排列
    """
    # 将用户问题转为向量
    query_embedding = client.get_embedding(query)

    # 计算与每张表的相似度
    scores = []
    for table_name, table_embedding in index.items():
        score = cosine_similarity(query_embedding, table_embedding)
        scores.append((table_name, score))

    # 按相似度降序排序，取 Top-K
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ==================== 5. 主程序：演示完整流程 ====================
if __name__ == "__main__":
    # 初始化 LLM 客户端（复用项目已有的配置）
    client = LLMClient()

    # 第一步：构建索引（将表描述向量化）
    table_index = build_table_index(client)

    # 第二步：用测试问题验证检索效果
    test_questions = [
        "欧洲市场最近三个月的销售额是多少？",
        "各产品线的毛利率",
        "上个月的研发费用和销售费用对比",
        "查询已完成订单的总数量",
        "按客户类型统计收入，需要换算成人民币",
    ]

    print("=" * 60)
    print("向量检索演示：根据用户问题召回相关表")
    print("=" * 60)

    for question in test_questions:
        print(f"\n问题：{question}")
        results = search_tables(question, table_index, client, top_k=3)
        print("召回结果：")
        for table_name, score in results:
            # 相似度 > 0.4 认为相关，用 ✓ 标记；否则用 · 标记
            marker = "✓" if score > 0.4 else "·"
            print(f"  {marker} {table_name:20s} 相似度: {score:.4f}")
