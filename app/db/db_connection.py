import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from app.services.logger import get_logger

# 加载.env文件中的环境变量
load_dotenv()


class DBConnection:
    """MySQL数据库连接管理类
    
    提供MySQL数据库连接管理和基本的数据库操作功能。
    支持从环境变量读取配置信息，实现数据库的连接、查询和结果获取等操作。
    包含完善的错误处理机制，确保数据库操作的安全性和可靠性。
    
    Attributes:
        host (str): 数据库主机地址
        user (str): 数据库用户名
        password (str): 数据库密码
        database (str): 数据库名称
        port (int): 数据库端口号
        connection: MySQL数据库连接对象
    """

    def __init__(self, assign_db=True):
        """初始化数据库连接参数
        
        从环境变量或传入参数获取数据库连接信息。
        优先使用传入的参数，如果未传入则使用环境变量中的配置。
        
        Args:
            config (dict, optional): 数据库连接配置字典，包含host、user、password、database、port等参数
        """
        self.logger = get_logger()
        self.logger.info("初始化数据库连接管理器...")
        self.config = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT'))
        }

        if assign_db:
            self.config['database'] = os.getenv('DB_DATABASE')
        self.connection = None
        # 修复属性访问方式
        self.logger.debug(
            f"数据库配置: host={self.config.get('host')}, port={self.config.get('port')}, database={self.config.get('database')}, user={self.config.get('user')}")


    def connect(self):
        """建立数据库连接

        尝试使用配置的参数建立MySQL数据库连接。
        如果连接失败，会记录错误信息。

        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.debug("尝试建立数据库连接...")
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                self.logger.info("数据库连接成功")
                return True
        except Error as e:
            self.logger.error(f"数据库连接失败: {e}")
            return False


    def disconnect(self):
        """关闭数据库连接

        安全地关闭当前的数据库连接。
        在关闭前会检查连接是否存在且处于连接状态。
        """
        if self.connection and self.connection.is_connected():
            self.logger.debug("关闭数据库连接")
            self.connection.close()


    def execute_query(self, query, params=None):
        """执行SQL查询

        执行给定的SQL查询语句，支持参数化查询以防止SQL注入。
        如果连接断时会自动尝试重新连接。

        Args:
            query (str): SQL查询语句
            params (tuple, optional): 查询参数

        Returns:
            cursor: 查询游标对象，如果执行失败则返回None

        Raises:
            Error: 当无法连接到数据库时抛出异常
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.logger.warning("数据库连接已断开，尝试重新连接")
                if not self.connect():
                    raise Error("无法连接到数据库")

            self.logger.debug(f"执行SQL查询: {query}, 参数: {params}")
            cursor = self.connection.cursor(buffered=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            self.logger.debug("SQL查询执行成功")
            return cursor
        except Error as e:
            self.logger.error(f"SQL查询执行失败: {e}")
            if 'cursor' in locals() and cursor:
                cursor.close()
            return None


    def fetch_all(self, query, params=None):
        """获取所有查询结果

        执行查询并返回所有匹配的结果行。

        Args:
            query (str): SQL查询语句
            params (tuple, optional): 查询参数

        Returns:
            list: 查询结果列表，如果查询失败则返回None
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.logger.warning("数据库连接已断开，尝试重新连接")
                if not self.connect():
                    raise Error("无法连接到数据库")

            self.logger.debug(f"执行fetch_all查询: {query}, 参数: {params}")
            cursor = self.execute_query(query, params)
            if cursor:
                result = cursor.fetchall()
                cursor.close()
                self.logger.debug(f"查询成功，返回{len(result)}条记录")
                return result
            return None
        except Error as e:
            self.logger.error(f"获取数据失败: {e}")
            return None


    def fetch_one(self, query, params=None):
        """获取单条查询结果

        执行查询并返回第一个匹配的结果行。

        Args:
            query (str): SQL查询语句
            params (tuple, optional): 查询参数

        Returns:
            tuple: 单条查询结果，如果查询失败或无结果则返回None
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.logger.warning("数据库连接已断开，尝试重新连接")
                if not self.connect():
                    raise Error("无法连接到数据库")

            self.logger.debug(f"执行fetch_one查询: {query}, 参数: {params}")
            cursor = self.execute_query(query, params)
            if cursor:
                result = cursor.fetchone()
                cursor.close()
                self.logger.debug(f"查询成功，返回结果: {result}")
                return result
            return None
        except Error as e:
            self.logger.error(f"获取数据失败: {e}")
            return None
