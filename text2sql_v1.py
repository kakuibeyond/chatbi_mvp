"""
第5课实战版本1：手写 Schema + Few-shot

在 Zero-shot 基础上注入手写 Schema 和 Few-shot 示例。
展示注入 Schema 和 Few-shot 示例后，LLM 生成 SQL 的准确性提升。

运行方式：
    uv run python text2sql_v1.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

import re
from openai import OpenAI


# ==================== 手写 Schema（完全靠人工整理）====================
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
"""


def build_prompt(user_question: str) -> str:
    """构造包含 Schema 和 Few-shot 的 Prompt"""
    prompt = f"""你是一个专业的 SQL 生成助手，擅长根据业务问题生成标准 MySQL 查询语句。

【数据库Schema】
{SCHEMA}

【示例】
{FEW_SHOT_EXAMPLES}

【用户问题】
{user_question}

【要求】
1. 只输出 SQL 语句，不需要解释
2. 使用标准 MySQL 语法
3. 确保字段名和表名与 Schema 一致
4. 如果涉及多表查询，使用 JOIN 连接
5. 收入口径统一使用 net_amount，成本口径使用 material_cost + labor_cost

请直接输出 SQL：
"""
    return prompt


def generate_sql(user_question: str) -> str:
    """调用 LLM 生成 SQL"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    prompt = build_prompt(user_question)

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000
    )

    raw_output = response.choices[0].message.content.strip()
    sql = re.sub(r'```sql|```', '', raw_output).strip()
    return sql


def run(question: str) -> None:
    """运行并输出生成的 SQL"""
    print(f"\n{'='*60}")
    print(f"问题：{question}")
    print(f"{'='*60}")
    sql = generate_sql(question)
    print(f"\n生成 SQL：\n{sql}")
    print("\n可将上述 SQL 复制到 MySQL 客户端执行验证。")


if __name__ == "__main__":
    questions = [
        "查询已完成订单的总数量",
        "按产品线统计总收入",
        "欧洲市场最近三个月的销售额是多少",
        "各产品线的毛利率是多少",
    ]
    for q in questions:
        run(q)
