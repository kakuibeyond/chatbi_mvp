"""
字段语义匹配模块

第 16 课：在表召回基础上，实现字段级的语义匹配。
使用 LangChain + ChromaDB（复用第 15 课基础设施），
为候选表的所有字段构建向量索引，结合业务规则实现混合匹配。

解决经典歧义：gross_amount vs net_amount、region vs country 等。

依赖安装：同第 15 课（langchain-openai, langchain-chroma, chromadb）
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import LLM_CONFIG
from table_retriever import retrieve_tables


# ==================== 字段描述数据 ====================
# 每个字段构建"字段名 + 所属表 + 数据类型 + 业务含义 + 枚举/示例"的结构化描述
FIELD_METADATA = {
    # ---- dim_customers ----
    "dim_customers.customer_id": {
        "table": "dim_customers",
        "field": "customer_id",
        "description": (
            "客户唯一标识（主键），INT 类型。"
            "用于关联 sales_orders 表的外键 customer_id。"
        ),
        "domain": "维度表",
    },
    "dim_customers.customer_name": {
        "table": "dim_customers",
        "field": "customer_name",
        "description": (
            "客户名称，VARCHAR(100)。"
            "示例：宝马集团、国家电网、特斯拉。"
            "用于查询特定客户的订单或收入。"
        ),
        "domain": "维度表",
    },
    "dim_customers.customer_type": {
        "table": "dim_customers",
        "field": "customer_type",
        "description": (
            "客户类型分类，VARCHAR(50)。"
            "枚举值：OEM整车厂 / 储能集成商 / 电网集团 / 工商业用户 / 换电运营商 / 经销商。"
            "用于按客户类型维度做收入、订单分布分析。"
        ),
        "domain": "维度表",
    },
    "dim_customers.industry": {
        "table": "dim_customers",
        "field": "industry",
        "description": (
            "客户所属行业，VARCHAR(50)。"
            "枚举值：交通 / 能源 / 工业 / 特种交通。"
            "用于按行业维度统计客户分布和收入占比。"
        ),
        "domain": "维度表",
    },
    "dim_customers.country": {
        "table": "dim_customers",
        "field": "country",
        "description": (
            "客户所在的具体国家，VARCHAR(50)。"
            "示例：Germany、United States、Japan、China。"
            "注意与 region（大区）的区别：country 是具体国家，region 是地区分组。"
            "当问题提到具体国家名称时使用此字段。"
        ),
        "domain": "维度表",
    },
    "dim_customers.region": {
        "table": "dim_customers",
        "field": "region",
        "description": (
            "客户所属的销售大区，VARCHAR(50)。"
            "枚举值：欧洲 / 北美 / 亚太 / 中东非洲 / 拉美。"
            "注意与 country（国家）的区别：region 是大区汇总维度。"
            "当问题说'某个市场/大区'时使用此字段，说具体国家名时用 country。"
        ),
        "domain": "维度表",
    },
    # ---- dim_products ----
    "dim_products.product_id": {
        "table": "dim_products",
        "field": "product_id",
        "description": (
            "产品唯一标识（主键），INT 类型。"
            "用于关联 sales_orders 表的外键 product_id。"
        ),
        "domain": "维度表",
    },
    "dim_products.product_name": {
        "table": "dim_products",
        "field": "product_name",
        "description": (
            "产品名称，VARCHAR(100)。"
            "示例：极氪001专用电池包、电网级液冷储能柜。"
            "用于查询特定产品的销量或收入。"
        ),
        "domain": "维度表",
    },
    "dim_products.product_line": {
        "table": "dim_products",
        "field": "product_line",
        "description": (
            "产品线分类，VARCHAR(50)。"
            "枚举值：动力电池-乘用车 / 动力电池-商用车 / 储能系统-电网级 / "
            "储能系统-工商业 / 电池材料与回收。"
            "用于按产品线维度分析收入、毛利。"
            "当用户提到'各产品线'或'按业务板块'时使用此字段。"
        ),
        "domain": "维度表",
    },
    "dim_products.category": {
        "table": "dim_products",
        "field": "category",
        "description": (
            "产品分类（细分品类），VARCHAR(50)。"
            "枚举值：高能量密度型 / 超快充型 / 混动专用型 / 低温适配型 / "
            "商用车标准型 / 电网级储能型 / 工商业储能型。"
            "比 product_line 更细的产品分类维度。"
        ),
        "domain": "维度表",
    },
    "dim_products.tech_route": {
        "table": "dim_products",
        "field": "tech_route",
        "description": (
            "技术路线，VARCHAR(50)。"
            "枚举值：三元锂 / 磷酸铁锂 / 钠离子 / 固态电池。"
            "用于按技术路线分析产品结构和成本差异。"
        ),
        "domain": "维度表",
    },
    "dim_products.standard_cost": {
        "table": "dim_products",
        "field": "standard_cost",
        "description": (
            "产品标准成本，DECIMAL(10,2)。"
            "= material_cost + labor_cost + 制造费用分摊。"
            "用于核算产品总成本。注意：毛利计算建议用 material_cost + labor_cost。"
        ),
        "domain": "维度表",
    },
    "dim_products.material_cost": {
        "table": "dim_products",
        "field": "material_cost",
        "description": (
            "材料成本，DECIMAL(10,2)。"
            "产品的原材料采购成本（正极材料、电解液、隔膜等）。"
            "毛利计算公式中的核心组成部分：毛利 = net_amount - (material_cost + labor_cost) * quantity。"
        ),
        "domain": "维度表",
    },
    "dim_products.labor_cost": {
        "table": "dim_products",
        "field": "labor_cost",
        "description": (
            "人工成本，DECIMAL(10,2)。"
            "产品的人工制造成本。"
            "毛利计算公式中的核心组成部分：毛利 = net_amount - (material_cost + labor_cost) * quantity。"
        ),
        "domain": "维度表",
    },
    # ---- sales_orders ----
    "sales_orders.order_id": {
        "table": "sales_orders",
        "field": "order_id",
        "description": "订单唯一标识（主键），BIGINT 类型。",
        "domain": "事实表",
    },
    "sales_orders.order_no": {
        "table": "sales_orders",
        "field": "order_no",
        "description": (
            "订单编号，VARCHAR(50)。业务编号，格式如 ORD-2026-001234。"
            "用于查询特定订单明细。"
        ),
        "domain": "事实表",
    },
    "sales_orders.customer_id": {
        "table": "sales_orders",
        "field": "customer_id",
        "description": (
            "客户外键，INT。关联 dim_customers.customer_id。"
            "当需要按客户维度分析时，通过此字段 JOIN dim_customers。"
        ),
        "domain": "事实表",
    },
    "sales_orders.product_id": {
        "table": "sales_orders",
        "field": "product_id",
        "description": (
            "产品外键，INT。关联 dim_products.product_id。"
            "当需要按产品维度分析时，通过此字段 JOIN dim_products。"
        ),
        "domain": "事实表",
    },
    "sales_orders.region": {
        "table": "sales_orders",
        "field": "region",
        "description": (
            "订单销售区域（冗余字段），VARCHAR(50)。"
            "枚举值：欧洲 / 北美 / 亚太 / 中东非洲 / 拉美。"
            "与 dim_customers.region 相同，冗余存储在订单表中便于直接筛选。"
            "当只需按大区过滤收入且不需要其他客户信息时，可直接使用此字段避免 JOIN。"
        ),
        "domain": "事实表",
    },
    "sales_orders.order_date": {
        "table": "sales_orders",
        "field": "order_date",
        "description": (
            "订单日期，DATE 类型。格式 YYYY-MM-DD。"
            "所有时间范围筛选（本月、上月、最近N个月、季度、年度）都基于此字段。"
            "也用于关联 exchange_rates.rate_date 进行汇率换算。"
        ),
        "domain": "事实表",
    },
    "sales_orders.order_status": {
        "table": "sales_orders",
        "field": "order_status",
        "description": (
            "订单状态，VARCHAR(20)。"
            "枚举值：completed（已完成）/ cancelled（已取消）/ pending（待处理）。"
            "统计收入、订单量等指标时，必须过滤 order_status = 'completed'。"
        ),
        "domain": "事实表",
    },
    "sales_orders.quantity": {
        "table": "sales_orders",
        "field": "quantity",
        "description": (
            "订单数量，DECIMAL(10,2)。单位为 MWh 或套数。"
            "用于统计销量、计算成本（cost * quantity）。"
        ),
        "domain": "事实表",
    },
    "sales_orders.unit_price": {
        "table": "sales_orders",
        "field": "unit_price",
        "description": (
            "单价（不含税），DECIMAL(10,2)。每 MWh 或每套的价格。"
            "用于分析定价策略、计算客单价。"
        ),
        "domain": "事实表",
    },
    "sales_orders.discount_amount": {
        "table": "sales_orders",
        "field": "discount_amount",
        "description": (
            "折扣金额，DECIMAL(10,2)。该订单的折扣优惠金额。"
            "用于分析折扣力度、计算实际售价。"
        ),
        "domain": "事实表",
    },
    "sales_orders.gross_amount": {
        "table": "sales_orders",
        "field": "gross_amount",
        "description": (
            "含税总额，DECIMAL(12,2)。包含增值税的订单总金额。"
            "注意：除非用户明确要求'含税金额'，否则不应使用此字段。"
            "日常说的'收入''销售额'统一使用 net_amount（不含税收入）。"
        ),
        "domain": "事实表",
    },
    "sales_orders.net_amount": {
        "table": "sales_orders",
        "field": "net_amount",
        "description": (
            "不含税收入（财务口径的销售额），DECIMAL(12,2)。"
            "这是业务中'收入''销售额''营业收入'的标准字段。"
            "毛利计算：毛利 = net_amount - (material_cost + labor_cost) * quantity。"
            "统计收入时必须同时过滤 order_status = 'completed'。"
        ),
        "domain": "事实表",
    },
    "sales_orders.currency": {
        "table": "sales_orders",
        "field": "currency",
        "description": (
            "订单币种，VARCHAR(10)。示例：CNY、USD、EUR、JPY。"
            "当涉及多币种收入汇总时，需通过 currency + order_date 关联 exchange_rates 表。"
        ),
        "domain": "事实表",
    },
    # ---- exchange_rates ----
    "exchange_rates.rate_date": {
        "table": "exchange_rates",
        "field": "rate_date",
        "description": (
            "汇率日期，DATE 类型。"
            "通过 sales_orders.order_date = exchange_rates.rate_date 关联。"
        ),
        "domain": "参考表",
    },
    "exchange_rates.currency": {
        "table": "exchange_rates",
        "field": "currency",
        "description": (
            "币种代码，VARCHAR(10)。示例：USD、EUR、JPY。"
            "通过 sales_orders.currency = exchange_rates.currency 关联。"
        ),
        "domain": "参考表",
    },
    "exchange_rates.rate_to_cny": {
        "table": "exchange_rates",
        "field": "rate_to_cny",
        "description": (
            "兑人民币汇率，DECIMAL(10,4)。"
            "换算公式：人民币金额 = 外币金额 * rate_to_cny。"
            "用于多币种收入统一折算为人民币。"
        ),
        "domain": "参考表",
    },
    # ---- finance_expenses ----
    "finance_expenses.expense_id": {
        "table": "finance_expenses",
        "field": "expense_id",
        "description": "费用记录唯一标识（主键），BIGINT 类型。",
        "domain": "事实表",
    },
    "finance_expenses.expense_date": {
        "table": "finance_expenses",
        "field": "expense_date",
        "description": (
            "费用日期，DATE 类型。格式 YYYY-MM-DD。"
            "用于按时间段筛选费用记录。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.department": {
        "table": "finance_expenses",
        "field": "department",
        "description": (
            "部门名称，VARCHAR(50)。"
            "示例：研发部、销售部、管理部。"
            "用于按部门维度分析费用。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.rd_expense": {
        "table": "finance_expenses",
        "field": "rd_expense",
        "description": (
            "研发费用，DECIMAL(12,2)。"
            "企业研发投入（电池技术研发、储能系统研发等）。"
            "新能源企业研发投入占比高，是重要的费用分析维度。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.selling_expense": {
        "table": "finance_expenses",
        "field": "selling_expense",
        "description": (
            "销售费用（总项），DECIMAL(12,2)。"
            "包含子项：marketing_expense + logistics_expense + warranty_expense。"
            "注意：汇总时不要重复计算！selling_expense 已包含其子项。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.admin_expense": {
        "table": "finance_expenses",
        "field": "admin_expense",
        "description": (
            "管理费用，DECIMAL(12,2)。"
            "行政管理相关费用。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.finance_expense": {
        "table": "finance_expenses",
        "field": "finance_expense",
        "description": (
            "财务费用，DECIMAL(12,2)。"
            "利息支出、汇兑损益等财务相关费用。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.marketing_expense": {
        "table": "finance_expenses",
        "field": "marketing_expense",
        "description": (
            "市场费用，DECIMAL(12,2)。"
            "属于 selling_expense 的子项。包括展会、广告、品牌推广等。"
            "注意：是销售费用的子项，不可与 selling_expense 加总。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.logistics_expense": {
        "table": "finance_expenses",
        "field": "logistics_expense",
        "description": (
            "物流费用，DECIMAL(12,2)。"
            "属于 selling_expense 的子项。电池产品运输、仓储等费用。"
            "注意：是销售费用的子项，不可与 selling_expense 加总。"
        ),
        "domain": "事实表",
    },
    "finance_expenses.warranty_expense": {
        "table": "finance_expenses",
        "field": "warranty_expense",
        "description": (
            "质保费用，DECIMAL(12,2)。"
            "属于 selling_expense 的子项。电池质保期内的维修、更换费用。"
            "注意：是销售费用的子项，不可与 selling_expense 加总。"
        ),
        "domain": "事实表",
    },
}


# ==================== 业务规则层 ====================
# 规则类型：whitelist（强制包含）、blacklist（强制排除）、conditional（条件触发）
BUSINESS_RULES = [
    # --- 收入口径规则 ---
    {
        "type": "whitelist",
        "trigger_keywords": ["收入", "销售额", "营业收入", "营收"],
        "force_include": ["sales_orders.net_amount"],
        "reason": "收入口径统一使用 net_amount（不含税）",
    },
    {
        "type": "blacklist",
        "trigger_keywords": ["收入", "销售额", "营业收入", "营收"],
        "force_exclude": ["sales_orders.gross_amount"],
        "reason": "除非明确要求含税，否则排除 gross_amount",
    },
    # --- 含税场景例外 ---
    {
        "type": "whitelist",
        "trigger_keywords": ["含税", "含税金额", "含税收入"],
        "force_include": ["sales_orders.gross_amount"],
        "reason": "用户明确要求含税时使用 gross_amount",
    },
    {
        "type": "blacklist",
        "trigger_keywords": ["含税", "含税金额", "含税收入"],
        "force_exclude": ["sales_orders.net_amount"],
        "reason": "用户明确要求含税时禁用 net_amount",
    },
    # --- 成本口径规则 ---
    {
        "type": "whitelist",
        "trigger_keywords": ["成本", "毛利", "利润"],
        "force_include": ["dim_products.material_cost", "dim_products.labor_cost"],
        "reason": "成本计算使用 material_cost + labor_cost",
    },
    # --- 订单过滤规则 ---
    {
        "type": "whitelist",
        "trigger_keywords": ["收入", "销售额", "订单量", "订单数", "客单价"],
        "force_include": ["sales_orders.order_status"],
        "reason": "收入类统计必须包含 order_status 用于过滤 completed",
    },
    # --- 汇率规则 ---
    {
        "type": "conditional",
        "trigger_keywords": ["人民币", "汇率", "换算", "折算", "统一币种"],
        "force_include": [
            "exchange_rates.rate_to_cny",
            "exchange_rates.rate_date",
            "exchange_rates.currency",
        ],
        "reason": "涉及汇率换算时必须引入 exchange_rates 表字段",
    },
    # --- 费用层级保护 ---
    {
        "type": "blacklist",
        "trigger_keywords": ["总费用", "期间费用合计", "费用汇总"],
        "force_exclude": [
            "finance_expenses.marketing_expense",
            "finance_expenses.logistics_expense",
            "finance_expenses.warranty_expense",
        ],
        "reason": "汇总费用时排除子项字段，避免重复计算",
    },
]


# ==================== ChromaDB 配置 ====================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db", "fields")


def _cosine_relevance_score_fn(distance: float) -> float:
    """
    将 ChromaDB 的 cosine 距离转换为余弦相似度。

    与 table_retriever.py 保持一致的分数转换逻辑。
    """
    return 1 - distance


def get_embeddings() -> OpenAIEmbeddings:
    """构建 LangChain OpenAI Embeddings 实例（复用第 15 课配置）"""
    return OpenAIEmbeddings(
        model=LLM_CONFIG["embedding_model"],
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"],
    )


def get_vectorstore() -> Chroma:
    """获取或创建字段描述的 ChromaDB 向量存储实例"""
    return Chroma(
        collection_name="field_descriptions",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
        relevance_score_fn=_cosine_relevance_score_fn,
    )


# ==================== 索引构建 ====================
def build_field_index(force_rebuild: bool = False) -> Chroma:
    """
    将所有字段描述向量化并写入 ChromaDB。

    Args:
        force_rebuild: 是否强制重建索引

    Returns:
        Chroma 向量存储实例
    """
    vectorstore = get_vectorstore()
    existing = vectorstore._collection.count()

    if existing > 0 and not force_rebuild:
        print(f"字段索引已存在（{existing} 条），跳过重建。如需重建请传入 force_rebuild=True")
        return vectorstore

    if force_rebuild and existing > 0:
        existing_ids = vectorstore._collection.get()["ids"]
        if existing_ids:
            vectorstore._collection.delete(ids=existing_ids)
        print("已清空旧字段索引数据")

    # 构建 Document 列表
    documents = []
    ids = []
    for field_key, meta in FIELD_METADATA.items():
        doc = Document(
            page_content=meta["description"],
            metadata={
                "table_name": meta["table"],
                "field_name": meta["field"],
                "field_key": field_key, # 如 "sales_orders.net_amount"
                "domain": meta["domain"],
            },
        )
        documents.append(doc)
        ids.append(field_key)

    # 写入向量数据库
    vectorstore.add_documents(documents, ids=ids)
    print(f"字段索引构建完成：{len(documents)} 个字段已写入 ChromaDB")
    print(f"持久化路径：{CHROMA_PERSIST_DIR}")
    return vectorstore


# ==================== 业务规则评估 ====================
def evaluate_rules(query: str) -> dict:
    """
    根据用户问题评估业务规则，返回强制包含和排除的字段列表。

    Args:
        query: 用户的自然语言问题

    Returns:
        {"force_include": [field_key, ...], "force_exclude": [field_key, ...]}
    """
    force_include = set()
    force_exclude = set()

    query_lower = query.lower()
    for rule in BUSINESS_RULES:
        # 检查触发关键词是否命中
        triggered = any(kw in query_lower for kw in rule["trigger_keywords"])
        if not triggered:
            continue

        if rule["type"] in ("whitelist", "conditional"):
            for field_key in rule.get("force_include", []):
                force_include.add(field_key)
        elif rule["type"] == "blacklist":
            for field_key in rule.get("force_exclude", []):
                force_exclude.add(field_key)

    # 白名单优先级高于黑名单（如果同一字段同时被 include 和 exclude，保留 include）
    force_exclude -= force_include

    return {
        "force_include": list(force_include),
        "force_exclude": list(force_exclude),
    }


# ==================== 混合匹配 ====================
def match_fields(
    query: str,
    candidate_tables: list[str] | None = None,
    top_k: int = 10,
    score_threshold: float = 0.15,
    rule_weight: float = 0.3,
) -> list[dict]:
    """
    混合匹配：向量相似度 + 业务规则，返回与问题最相关的字段。

    评分公式：final_score = (1 - rule_weight) * embedding_score + rule_weight * rule_score
    其中 rule_score: 强制包含 = 1.0, 强制排除 = -1.0, 无规则 = 0.0

    Args:
        query: 用户的自然语言问题
        candidate_tables: 候选表列表（通常来自 table_retriever 的结果）。
                         如果为 None，则在所有字段中搜索。
        top_k: 返回前 K 个字段
        score_threshold: 最终得分阈值
        rule_weight: 规则分数的权重（0~1）

    Returns:
        [{"field_key": str, "table": str, "field": str, "score": float,
          "embedding_score": float, "rule_applied": str|None}]
    """
    vectorstore = get_vectorstore()

    # 1. 构建过滤条件：限定在候选表范围内
    search_kwargs = {"k": min(top_k * 3, 30)}  # 多检索一些，后续再过滤
    if candidate_tables and len(candidate_tables) == 1:
        search_kwargs["filter"] = {"table_name": candidate_tables[0]}
    elif candidate_tables and len(candidate_tables) > 1:
        search_kwargs["filter"] = {"table_name": {"$in": candidate_tables}}

    # 2. 向量检索
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query, **search_kwargs
    )

    # 3. 评估业务规则
    rule_result = evaluate_rules(query)
    force_include = set(rule_result["force_include"])
    force_exclude = set(rule_result["force_exclude"])

    # 4. 混合评分
    scored_fields = []
    seen_keys = set()

    for doc, embedding_score in results_with_scores:
        field_key = doc.metadata["field_key"]
        if field_key in seen_keys:
            continue
        seen_keys.add(field_key)

        # 如果有表限制且字段不在候选表内，跳过
        if candidate_tables and doc.metadata["table_name"] not in candidate_tables:
            continue

        # 计算规则分数
        rule_score = 0.0
        rule_applied = None
        if field_key in force_include:
            rule_score = 1.0
            rule_applied = "强制包含"
        elif field_key in force_exclude:
            rule_score = -1.0
            rule_applied = "强制排除"

        # 混合分数
        final_score = (1 - rule_weight) * embedding_score + rule_weight * rule_score

        scored_fields.append({
            "field_key": field_key,
            "table": doc.metadata["table_name"],
            "field": doc.metadata["field_name"],
            "score": round(final_score, 4),
            "embedding_score": round(embedding_score, 4),
            "rule_applied": rule_applied,
            "description": doc.page_content,
        })

    # 5. 补充强制包含但未被向量检索命中的字段
    for field_key in force_include:
        if field_key not in seen_keys:
            # 检查是否在候选表范围内
            meta = FIELD_METADATA.get(field_key)
            if meta and (not candidate_tables or meta["table"] in candidate_tables):
                scored_fields.append({
                    "field_key": field_key,
                    "table": meta["table"],
                    "field": meta["field"],
                    "score": round(rule_weight * 1.0, 4),  # 纯规则分
                    "embedding_score": 0.0,
                    "rule_applied": "强制包含（补充）",
                    "description": meta["description"],
                })

    # 6. 排序 + 过滤
    scored_fields.sort(key=lambda x: x["score"], reverse=True)
    # 移除被强制排除且分数为负的字段
    scored_fields = [f for f in scored_fields if f["score"] >= score_threshold]
    return scored_fields[:top_k]


# ==================== 集成接口：表召回 + 字段匹配 ====================
def retrieve_schema(
    query: str,
    table_top_k: int = 3,
    field_top_k: int = 10,
    table_threshold: float = 0.2,
    field_threshold: float = 0.15,
) -> dict:
    """
    表+字段联合召回：先用 table_retriever 召回相关表，
    再在候选表范围内做字段匹配。

    Args:
        query: 用户的自然语言问题
        table_top_k: 表召回数量
        field_top_k: 字段匹配数量
        table_threshold: 表召回阈值
        field_threshold: 字段匹配阈值

    Returns:
        {
            "tables": [{"table_name": str, "score": float, ...}],
            "fields": [{"field_key": str, "table": str, "field": str, "score": float, ...}],
            "schema_snippet": str  # 精简的 Schema 文本片段
        }
    """
    # 1. 表召回
    tables = retrieve_tables(query, top_k=table_top_k, score_threshold=table_threshold)
    candidate_table_names = [t["table_name"] for t in tables]

    # 2. 字段匹配（限定在召回的表范围内）
    fields = match_fields(
        query,
        candidate_tables=candidate_table_names,
        top_k=field_top_k,
        score_threshold=field_threshold,
    )

    # 3. 组装精简 Schema 片段
    schema_snippet = _build_schema_snippet(tables, fields)

    return {
        "tables": tables,
        "fields": fields,
        "schema_snippet": schema_snippet,
    }


def _build_schema_snippet(tables: list[dict], fields: list[dict]) -> str:
    """根据召回的表和字段，生成精简的 Schema 文本"""
    # 按表分组字段
    table_fields: dict[str, list[str]] = {}
    for t in tables:
        table_fields[t["table_name"]] = []

    for f in fields:
        if f["table"] in table_fields:
            table_fields[f["table"]].append(f["field"])

    # 生成 Schema 片段
    lines = []
    for table_name, field_list in table_fields.items():
        if field_list:
            fields_str = ", ".join(field_list)
            lines.append(f"表：{table_name}（相关字段：{fields_str}）")
        else:
            lines.append(f"表：{table_name}")

    return "\n".join(lines) if lines else "（未召回相关表）"


# ==================== 主程序：演示 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("字段语义匹配演示（向量相似度 + 业务规则混合匹配）")
    print("=" * 60)

    # 第一步：构建字段索引
    print("\n--- 构建字段向量索引 ---")
    build_field_index(force_rebuild=False)

    # 第二步：经典歧义案例测试
    test_cases = [
        {
            "question": "查询上个月各大区的销售额",
            "focus": "应选 net_amount 而非 gross_amount，应选 region 而非 country",
        },
        {
            "question": "各产品线的毛利率",
            "focus": "应包含 net_amount + material_cost + labor_cost + quantity",
        },
        {
            "question": "按客户类型统计含税收入",
            "focus": "明确说含税，应选 gross_amount",
        },
    ]

    print("\n--- 字段匹配测试 ---")
    for case in test_cases:
        q = case["question"]
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        print(f"关注点：{case['focus']}")
        print("-" * 50)

        result = retrieve_schema(q)

        print(f"召回表：{[t['table_name'] for t in result['tables']]}")
        print(f"匹配字段（Top-5）：")
        for f in result["fields"][:5]:
            rule_tag = f" [{f['rule_applied']}]" if f["rule_applied"] else ""
            print(
                f"  {f['field_key']:40s} "
                f"总分:{f['score']:.3f} "
                f"向量:{f['embedding_score']:.3f}"
                f"{rule_tag}"
            )
        print(f"\n精简Schema:\n  {result['schema_snippet']}")

    # 第三步：规则评估演示
    print(f"\n\n{'='*60}")
    print("业务规则评估演示")
    print("=" * 60)
    rule_tests = ["查询各大区的销售额", "含税收入统计", "利润和成本分析"]
    for q in rule_tests:
        rules = evaluate_rules(q)
        print(f"\n问题：{q}")
        if rules["force_include"]:
            print(f"  强制包含：{rules['force_include']}")
        if rules["force_exclude"]:
            print(f"  强制排除：{rules['force_exclude']}")
