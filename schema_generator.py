"""
Schema 自动生成工具

从 MySQL 数据库连接中读取表结构，按固定模板生成 Schema 字符串，
替代完全手写的方式。

运行方式：
    uv run python schema_generator.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

import pymysql


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "chatbi_mvp"),
    "charset": "utf8mb4"
}


def get_tables(cursor) -> list[str]:
    """获取数据库中的所有表名"""
    cursor.execute("SHOW TABLES")
    return [row[0] for row in cursor.fetchall()]


def get_columns(cursor, table_name: str) -> list[dict]:
    """获取指定表的所有字段信息"""
    cursor.execute(f"SHOW FULL COLUMNS FROM `{table_name}`")
    columns = []
    for row in cursor.fetchall():
        columns.append({
            "name": row[0],
            "type": row[1],
            "comment": row[8] or ""
        })
    return columns


def get_foreign_keys(cursor, table_name: str) -> list[dict]:
    """获取指定表的外键信息"""
    cursor.execute(f"""
        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = '{table_name}'
          AND REFERENCED_TABLE_NAME IS NOT NULL
    """)
    fks = []
    for row in cursor.fetchall():
        fks.append({
            "column": row[0],
            "ref_table": row[1],
            "ref_column": row[2]
        })
    return fks


def build_schema_string(cursor, table_name: str) -> str:
    """为单张表生成 Schema 描述字符串"""
    columns = get_columns(cursor, table_name)
    fks = get_foreign_keys(cursor, table_name)
    fk_map = {fk["column"]: fk for fk in fks}

    lines = [f"表：{table_name}"]
    for col in columns:
        name = col["name"]
        col_type = col["type"]
        comment = col["comment"]

        parts = [f"- {name} {col_type}"]
        if name in fk_map:
            fk = fk_map[name]
            parts.append(f"外键 → {fk['ref_table']}.{fk['ref_column']}")
        elif comment:
            parts.append(comment)

        lines.append(" ".join(parts))

    return "\n".join(lines)


def generate_schema() -> str:
    """连接数据库，生成所有表的 Schema 描述"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            tables = get_tables(cursor)
            schemas = []
            for table in tables:
                schema_str = build_schema_string(cursor, table)
                schemas.append(schema_str)
            return "\n\n".join(schemas)
    finally:
        conn.close()


def main():
    """主入口：输出生成的 Schema 字符串"""
    print("=" * 60)
    print("正在从数据库自动生成 Schema...")
    print("=" * 60)

    schema = generate_schema()

    print("\n生成的 Schema 字符串：\n")
    print(schema)

    print("\n" + "=" * 60)
    print("可将上述 Schema 字符串复制到 Prompt 中使用。")
    print("=" * 60)


if __name__ == "__main__":
    main()
