import os
import time
from pathlib import Path
from dotenv import load_dotenv
from app.db.db_connection import DBConnection
from app.db.redis_connection import RedisConnection
from app.services.logger import get_logger

# 加载环境变量
load_dotenv()


class VideoSystemInitializer:
    """
    视频系统初始化器，负责系统的初始化工作

    该类负责完成系统启动时的初始化工作，包括：
    1. 数据库连接和初始化
    2. 数据表创建和维护
    3. Redis缓存预热
    """

    def __init__(self):
        """初始化系统初始化器
        """
        self.logger = get_logger()
        self.logger.info("初始化系统初始化器")

        self.database_name = os.getenv('DB_DATABASE', 'video_duplicate_check')
        self.table_name = os.getenv('DB_TABLENAME', 'videos')

        # 首先尝试连接MySQL服务器（不指定数据库）
        self.db = DBConnection(False)

        # 设置数据库
        self.setup_database()

        self.db = DBConnection(True)

        # 设置数据表
        self.setup_table()

        self.redis = RedisConnection()

        # 执行缓存预热
        self.warm_up_cache()
        self.logger.info("系统初始化器初始化完成")

    def setup_database(self):
        """创建必要的数据库

        该方法完成以下任务：
        1. 检查数据库连接
        2. 检查并创建数据库（如果不存在）

        Raises:
            ConnectionError: 数据库连接失败时抛出
        """
        # 检查数据库是否存在
        check_db_query = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s"
        result = self.db.fetch_one(check_db_query, (self.database_name,))

        if not result:
            self.logger.info(f"数据库 {self.database_name} 不存在，正在创建...")
            create_db_query = f"CREATE DATABASE {self.database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            self.db.execute_query(create_db_query)
            self.logger.info(f"数据库 {self.database_name} 创建成功")
        else:
            self.logger.info(f"数据库 {self.database_name} 已存在")

    def setup_table(self):
        """创建必要的数据表，并执行缓存预热

        该方法完成以下任务：
        1. 检查并创建视频信息表（如果不存在）
        2. 执行缓存预热

        Raises:
            ConnectionError: 数据库连接失败时抛出
        """
        # 检查数据库连接状态
        if not self.db.connection or not self.db.connection.is_connected():
            self.logger.error("数据库连接已断开，尝试重新连接")
            if not self.db.connect():
                raise ConnectionError("无法连接到数据库")

        # 确保使用正确的数据库
        self.db.connection.database = self.database_name

        self.logger.info("检查数据表是否存在")
        check_table_query = """
        SELECT COUNT(*)
        FROM information_schema.tables 
        WHERE table_schema = %s 
        AND table_name = %s
        """
        result = self.db.fetch_one(check_table_query, (self.database_name, self.table_name))

        # 只在表不存在时创建表
        if result[0] == 0:
            try:
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INT unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
                    video_id INT unsigned NOT NULL COMMENT '业务系统视频-ID',
                    industry_code VARCHAR(64) NOT NULL DEFAULT '' COMMENT '业务系统视频分类-Code',
                    file_hash VARCHAR(64) NOT NULL DEFAULT '' COMMENT '文件内容哈希',
                    feature_hash TEXT COMMENT '特征哈希',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX (file_hash),
                    INDEX idx_feature_hash (feature_hash(64))
                )
                """
                if self.db.execute_query(create_table_query):
                    self.logger.info("数据表创建成功")
                else:
                    self.logger.info("数据表创建失败")
            except Exception as e:
                self.logger.error(f"创建数据表失败: {e}")
                raise
        else:
            self.logger.info("数据库表已存在")

    def warm_up_cache(self):
        """
        缓存预热优化版（使用游标分页）
        缓存预热：将数据库中的视频哈希数据加载到Redis

        该方法通过批量加载方式将数据库中的视频哈希数据预加载到Redis缓存中，
        以提高后续查询性能。使用基于主键ID的游标分页方式，避免大数据量时的性能问题。
        """
        if not self.redis.cache_enabled:
            self.logger.info("缓存已禁用，跳过缓存预热")
            return

        # 检查Redis中是否已存在缓存数据
        file_hash_keys = self.redis.keys("file_hash:*")
        feature_hash_keys = self.redis.keys("feature_hash:*")

        if file_hash_keys or feature_hash_keys:
            self.logger.info("Redis中已存在缓存数据，跳过缓存预热")
            return

        self.logger.info("开始缓存预热...")
        batch_size = 2000  # 根据服务器性能调整批次大小
        last_id = 0
        total_loaded = 0
        start_time = time.time()

        try:
            while True:
                # 使用覆盖索引优化查询
                query = f"SELECT id, video_id, industry_code, file_hash, feature_hash FROM {self.table_name} WHERE id > %s ORDER BY id LIMIT %s"
                params = (last_id, batch_size)

                # 使用流式游标获取数据
                cursor = self.db.connection.cursor()
                cursor.execute(query, params)

                batch_count = 0
                while True:
                    # 分批获取数据减少内存占用
                    rows = cursor.fetchmany(size=500)
                    if not rows:
                        break

                    # 使用pipeline批量操作
                    pipe = self.redis.pipeline()
                    for row in rows:
                        id, video_id, industry_code, file_hash, feature_hash = row
                        # 使用JSON格式存储video_id和industry_code
                        cache_value = f"{{\"video_id\":{video_id},\"industry_code\":\"{industry_code}\"}}"
                        pipe.set(f"file_hash:{file_hash}", cache_value)
                        pipe.set(f"feature_hash:{feature_hash}", cache_value)
                        last_id = id
                        batch_count += 1

                    # 执行批量操作
                    pipe.execute()
                    total_loaded += batch_count
                    self.logger.debug(f"已加载批次: {batch_count} 条，总计: {total_loaded}")

                cursor.close()

                # 提前退出条件
                if batch_count < batch_size:
                    break

        except Exception as e:
            self.logger.error(f"缓存预热失败: {e}")
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()

        duration = time.time() - start_time
        self.logger.info(f"缓存预热完成，共加载 {total_loaded} 条记录，耗时 {duration:.2f} 秒")
