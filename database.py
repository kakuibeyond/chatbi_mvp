"""
数据库模块

负责数据库连接、SQL 执行和结果获取。
将数据库操作封装为独立模块，便于后续扩展（如连接池、读写分离等）。
"""

import pymysql
from typing import Tuple, List
from config import DB_CONFIG


class DatabaseClient:
    """MySQL 数据库客户端"""

    def __init__(self):
        self.config = DB_CONFIG

    def execute(self, sql: str) -> Tuple[List[str], List[tuple]]:
        """
        执行 SQL 并返回结果

        Args:
            sql: 待执行的 SQL 语句

        Returns:
            (列名列表, 结果行列表)
        """
        conn = pymysql.connect(**self.config)
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                results = cursor.fetchall()
                return columns, results
        finally:
            conn.close()

    def validate_connection(self) -> bool:
        """验证数据库连接是否正常"""
        try:
            conn = pymysql.connect(**self.config)
            conn.close()
            return True
        except Exception:
            return False
