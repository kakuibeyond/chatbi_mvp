"""
Prompt 构造模块

负责将 Schema 信息、Few-shot 示例和用户问题组装为完整 Prompt。
Schema 和示例在此集中维护，便于后续课程中动态扩展。

第7课增强：新增 RULES（业务规则注入层）与 ERROR_GUARDS（错误防护层），
通过 build_prompt 的可选参数控制是否注入，实现 Prompt 策略的灵活切换。
"""


# ==================== Schema 定义 ====================
SCHEMA = """
表：dim_customers（客户维度表）
- customer_id INT 主键
- customer_name VARCHAR(100) 客户名称
- customer_type VARCHAR(50) 客户类型：OEM整车厂 / 储能集成商 / 电网集团 / 工商业用户 / 换电运营商 / 经销商
- industry VARCHAR(50) 客户行业：交通 / 能源 / 工业 / 特种交通
- country VARCHAR(50) 具体国家，如 Germany
- region VARCHAR(50) 大区，如 欧洲、北美

表：dim_products（产品维度表）
- product_id INT 主键
- product_name VARCHAR(100) 产品名称
- product_line VARCHAR(50) 产品线：动力电池-乘用车 / 动力电池-商用车 / 储能系统-电网级 / 储能系统-工商业 / 电池材料与回收
- category VARCHAR(50) 产品分类：高能量密度型 / 超快充型 / 混动专用型 / 低温适配型 / 商用车标准型 / 电网级储能型 / 工商业储能型
- tech_route VARCHAR(50) 技术路线：三元锂 / 磷酸铁锂 / 钠离子 / 固态电池
- standard_cost DECIMAL(10,2) 标准成本
- material_cost DECIMAL(10,2) 材料成本
- labor_cost DECIMAL(10,2) 人工成本

表：sales_orders（销售订单表）
- order_id BIGINT 主键
- order_no VARCHAR(50) 订单编号
- customer_id INT 外键 → dim_customers.customer_id
- product_id INT 外键 → dim_products.product_id
- region VARCHAR(50) 销售区域
- order_date DATE 订单日期
- order_status VARCHAR(20) 订单状态：completed / cancelled / pending
- quantity DECIMAL(10,2) 数量（MWh 或套数）
- unit_price DECIMAL(10,2) 单价（每 MWh 或每套价格，不含税）
- discount_amount DECIMAL(10,2) 折扣金额
- gross_amount DECIMAL(12,2) 含税总额
- net_amount DECIMAL(12,2) 不含税收入（财务口径的销售额）
- currency VARCHAR(10) 币种

表：exchange_rates（汇率表）
- rate_date DATE 日期
- currency VARCHAR(10) 币种
- rate_to_cny DECIMAL(10,4) 兑人民币汇率

表：finance_expenses（费用表）
- expense_id BIGINT 主键
- expense_date DATE 费用日期
- department VARCHAR(50) 部门
- rd_expense DECIMAL(12,2) 研发费用（新能源企业研发投入大）
- selling_expense DECIMAL(12,2) 销售费用
- admin_expense DECIMAL(12,2) 管理费用
- finance_expense DECIMAL(12,2) 财务费用
- marketing_expense DECIMAL(12,2) 市场费用（属于销售费用子项）
- logistics_expense DECIMAL(12,2) 物流费用
- warranty_expense DECIMAL(12,2) 质保费用
"""


# ==================== Few-shot 示例 ====================
FEW_SHOT_EXAMPLES = """
示例1：
问题：查询已完成订单的总数量
SQL：SELECT COUNT(*) FROM sales_orders WHERE order_status = 'completed';

示例2：
问题：按客户类型统计订单数量
SQL：SELECT c.customer_type, COUNT(*) AS order_count FROM sales_orders o JOIN dim_customers c ON o.customer_id = c.customer_id WHERE o.order_status = 'completed' GROUP BY c.customer_type;

示例3：
问题：查询2026年第一季度的总费用
SQL：SELECT SUM(rd_expense + selling_expense + admin_expense + finance_expense) AS total_expense FROM finance_expenses WHERE expense_date >= '2026-01-01' AND expense_date < '2026-04-01';

示例4：
问题：查看最近三个月各月销售收入
SQL：SELECT
    DATE_FORMAT(o.order_date, '%Y-%m-01') AS month,
    SUM(o.net_amount * r.rate_to_cny) AS revenue_cny
FROM sales_orders o
JOIN exchange_rates r
  ON o.order_date = r.rate_date
 AND o.currency = r.currency
WHERE o.order_status = 'completed'
  AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
  AND o.order_date < CURDATE() + INTERVAL 1 DAY
GROUP BY DATE_FORMAT(o.order_date, '%Y-%m-01')
ORDER BY month;
"""


# ==================== 规则注入层（第7课新增）====================
RULES = """
【关键业务规则】
1. 收入口径："销售额""收入"均指不含税收入，统一使用 sales_orders.net_amount，禁止使用 gross_amount
2. 成本口径："成本"指实际销售成本，计算公式为 dim_products.material_cost + dim_products.labor_cost
3. 订单统计范围：统计收入、订单量、客单价等指标时，必须过滤 order_status = 'completed'，排除 cancelled 和 pending
4. 汇率转换：涉及多币种收入汇总时，必须通过 order_date 和 currency 关联 exchange_rates 表，使用 rate_to_cny 折算为人民币
5. 时间范围："最近N个月"使用 DATE_SUB(CURDATE(), INTERVAL N MONTH) 作为起始边界；"本月"指当月1日至当前日期
6. 费用层级：selling_expense 是销售费用总项，包含 marketing_expense、logistics_expense、warranty_expense，汇总时不得重复计算
7. 毛利计算：毛利 = net_amount - (material_cost + labor_cost) * quantity
8. 不要引用 Schema 中不存在的字段，例如 channel
"""


# ==================== 错误防护层（第7课新增）====================
ERROR_GUARDS = """
【常见错误防护】
- 字段选择：确认金额字段是 net_amount（不含税）还是 gross_amount（含税），除非明确要求"含税"，否则一律用 net_amount
- Join 遗漏：只要查询涉及"收入"且存在 currency 字段，必须关联 exchange_rates 表做汇率转换
- 过滤遗漏：所有收入类统计必须包含 WHERE order_status = 'completed'
- 时间边界：使用 >= 和 < 组合表示闭开区间；“最近N个月”按 DATE_SUB(CURDATE(), INTERVAL N MONTH) 处理
- 聚合维度：GROUP BY 字段必须与 SELECT 中的非聚合字段完全一致
- 字段合法性：不要输出 Schema 中不存在的字段；如果问题里出现未建模维度，优先回退到产品线、区域、客户、月份等已有维度
"""


# ==================== Prompt 构造函数 ====================
def build_prompt(
    user_question: str,
    use_few_shot: bool = True,
    use_rules: bool = False,
    use_guards: bool = False,
    indicator_knowledge: str = "",
    use_schema_linking: bool = False,
) -> tuple[str, str]:
    """
    构造发送给 LLM 的 Prompt

    Args:
        user_question: 用户的自然语言问题
        use_few_shot: 是否使用 Few-shot 示例
        use_rules: 是否注入业务规则层（第7课新增）
        use_guards: 是否注入错误防护层（第7课新增）
        indicator_knowledge: 指标知识文本块（第9课新增）
        use_schema_linking: 是否使用动态 Schema Linking（第18课新增）
                           为 True 时调用 schema_linker 动态生成精简 Schema，
                           失败时自动回退到全量 Schema。

    Returns:
        (system_message, user_message)
    """
    system_msg = "你是一个专业的 SQL 生成助手，擅长根据业务问题生成标准 MySQL 查询语句。"
    if use_rules or use_guards:
        system_msg = "你是一个专业的 SQL 生成助手，擅长根据业务问题生成标准 MySQL 查询语句。请严格遵守给定的业务规则，避免常见错误。"

    # 第18课新增：动态 Schema Linking（失败时自动回退全量 Schema）
    schema_text = SCHEMA
    if use_schema_linking:
        try:
            from schema_linker import build_dynamic_prompt_schema
            dynamic_schema = build_dynamic_prompt_schema(user_question)
            if dynamic_schema:
                schema_text = dynamic_schema
        except Exception:
            pass  # 回退到全量 SCHEMA

    prompt = f"""【数据库Schema】
{schema_text}
"""
    if use_rules:
        prompt += f"""
{RULES}
"""

    if use_few_shot:
        prompt += f"""
【示例】
{FEW_SHOT_EXAMPLES}
"""

    if use_guards:
        prompt += f"""
{ERROR_GUARDS}
"""

    if indicator_knowledge:
        prompt += f"""
{indicator_knowledge}
"""

    prompt += f"""
【用户问题】
{user_question}

【要求】
1. 只输出 SQL 语句，不需要解释
2. 使用标准 MySQL 语法
3. 确保字段名和表名与 Schema 一致
4. 如果涉及多表查询，使用 JOIN 连接
5. 收入口径统一使用 net_amount，成本口径使用 material_cost + labor_cost
6. 统计销售额时，需要按订单日期的汇率转换为人民币
7. SQL 必须完整闭合，CTE、SELECT、GROUP BY、ORDER BY 不得省略，不要输出截断的半句 SQL
"""
    if use_rules or use_guards:
        prompt += """8. 优先遵循【关键业务规则】和【常见错误防护】中的约束
"""

    prompt += """
请直接输出 SQL：
"""
    return system_msg, prompt
