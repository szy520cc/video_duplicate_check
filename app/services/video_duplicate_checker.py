import os
import argparse
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from app.db.db_connection import DBConnection
from app.db.redis_connection import RedisConnection
from app.services.video_feature_manager import VideoFeatureManager
from app.services.logger import get_logger
from app.services.init import VideoSystemInitializer

# 加载环境变量
load_dotenv()

# 初始化系统
VideoSystemInitializer()


class VideoDuplicateChecker:
    """视频重复检查器，负责视频的查重、添加和管理

    该类实现了基于特征匹配的视频重复检测系统，主要功能包括：
    1. 视频特征提取和存储
    2. 基于多维特征的相似度计算
    3. 数据库管理和缓存优化
    4. 批量视频处理

    Attributes:
        db_config (dict): 数据库连接配置
        similarity_threshold (float): 判定视频重复的相似度阈值
    """

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __init__(self):
        """初始化视频重复检查器

        Args:
            db_config (dict, optional): 数据库连接配置，默认从环境变量读取
            
        Raises:
            ConnectionError: 数据库或Redis连接失败时抛出
        """
        self.logger = get_logger()
        self.logger.info("初始化视频重复检查器...")
        try:
            self.db = DBConnection()
            if not self.db.connect():
                raise ConnectionError("无法连接到数据库")
            # 从环境变量获取表名
            self.table_name = os.getenv('DB_TABLENAME', 'videos')
            self.logger.debug(f"从环境变量读取表名: {self.table_name}")
            self.redis = RedisConnection()
            if not self.redis.connect():
                raise ConnectionError("无法连接到Redis")
                
            self.feature_manager = VideoFeatureManager()
            # 从环境变量读取相似度阈值，并确保特征管理器使用相同的阈值
            self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.50'))
            self.feature_manager.similarity_threshold = self.similarity_threshold
            self.logger.debug(f"从环境变量读取相似度阈值: {self.similarity_threshold}")
            self.logger.info("初始化视频重复检查器完成")
        except Exception as e:
            self.logger.error(f"初始化视频重复检查器失败: {str(e)}")
            raise ConnectionError(f"初始化失败: {str(e)}")




    def check_video(self, video_path):
        """检查视频是否与数据库中的视频重复

        该方法通过以下步骤检查视频重复：
        1. 计算视频文件哈希值
        2. 检查Redis缓存中是否存在完全相同的视频
        3. 检查数据库中是否存在完全相同的视频
        4. 提取视频特征并进行相似度比对

        Args:
            video_path (str): 待检查视频的文件路径

        Returns:
            tuple: (is_duplicate, duplicate_info, similarity)
            - is_duplicate (bool): 是否检测到重复
            - duplicate_info (str): 重复视频的信息（JSON字符串）
            - similarity (float): 相似度值

        Raises:
            FileNotFoundError: 视频文件不存在时抛出
        """
        self.logger.info(f"开始检查视频: {video_path}")

        if not Path(video_path).exists():
            self.logger.error(f"视频文件不存在: {video_path}")
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 计算文件哈希值
        self.logger.debug("开始计算文件哈希值...")
        file_hash = self.feature_manager.calculate_file_hash(video_path)
        self.logger.debug("文件哈希值计算完成")

        # 只在启用缓存时检查Redis
        if self.redis.cache_enabled:
            self.logger.debug("检查Redis缓存...")
            # 首先从Redis缓存中检查是否存在相同的文件
            cached_data = self.redis.get(f"file_hash:{file_hash}")
            if cached_data:
                self.logger.info("在Redis缓存中找到匹配文件")
                return True, cached_data, 1.0

        # 如果缓存中没有，再检查数据库
        self.logger.debug("在数据库中查找相同文件...")
        file_check_query = f"SELECT video_id, industry_code FROM {self.table_name} WHERE file_hash = %s"
        result = self.db.fetch_one(file_check_query, (file_hash,))
        if result:
            self.logger.info("在数据库中找到完全相同的文件")
            stored_video_id, stored_industry_code = result
            # 构造统一的JSON格式返回值
            duplicate_info = json.dumps({"video_id": stored_video_id, "industry_code": stored_industry_code})
            # 更新缓存
            self.redis.set(f"file_hash:{file_hash}", duplicate_info)
            return True, duplicate_info, 1.0

        # 提取新视频特征
        self.logger.debug("开始提取视频特征...")
        try:
            features = self.feature_manager.extract_video_features(video_path)
            feature_hash = self.feature_manager.calculate_feature_hash(features)
            self.logger.debug("视频特征提取完成")
        except Exception as e:
            self.logger.error(f"视频特征提取失败: {str(e)}")
            self.logger.error(f"视频路径: {video_path}")
            self.logger.error(f"错误详情: {e.__class__.__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                self.logger.error(f"堆栈跟踪:\n{traceback.format_tb(e.__traceback__)}")
            raise ValueError(f"视频特征提取失败: {str(e)}")

        # 进行视频相似度比对
        self.logger.debug("开始进行视频相似度比对...")
        max_similarity = 0.0
        duplicate_info = None

        # 如果启用了缓存，先从Redis缓存中获取特征哈希进行比对
        if self.redis.cache_enabled:
            self.logger.debug("从Redis缓存中获取特征哈希进行比对...")
            # 获取所有以"feature_hash:"为前缀的键
            cache_keys = self.redis.keys("feature_hash:*")

            if cache_keys:
                self.logger.debug(f"找到{len(cache_keys)}个缓存的特征哈希")
                for key in cache_keys:
                    try:
                        # 从键名中提取特征哈希值（去掉"feature_hash:"前缀）
                        length_of_prefix = len("feature_hash:")
                        stored_feature_hash = key[length_of_prefix:]
                        cached_data = self.redis.get(key)
                        stored_info = json.loads(cached_data)

                        # 使用增强的相似度计算方法
                        is_duplicate, similarity = self.feature_manager.is_duplicate(
                            feature_hash,
                            stored_feature_hash,
                            self.similarity_threshold
                        )

                        self.logger.debug(
                            f"缓存比对结果: 视频 {video_path} 与 video_id:{stored_info['video_id']} 的相似度为 {similarity:.4f}, 是否重复: {is_duplicate}")

                        if similarity > max_similarity:
                            max_similarity = similarity
                            duplicate_info = cached_data

                        if is_duplicate:
                            self.logger.info(f"在缓存中找到相似视频，相似度: {similarity:.2%}")
                            return True, duplicate_info, similarity
                    except Exception as e:
                        self.logger.error(f"从缓存计算相似度时出错: {e}")

                self.logger.debug("缓存中未找到相似度超过阈值的视频，继续从数据库查询...")
            else:
                self.logger.debug("缓存中没有特征哈希数据，将从数据库查询...")

        # 如果缓存中没有找到匹配或缓存未启用，从数据库中获取视频特征哈希进行比对
        batch_size = 1000
        offset = 0

        while True:
            query = f"SELECT id, video_id, industry_code, feature_hash FROM {self.table_name} LIMIT %s OFFSET %s"
            result = self.db.fetch_all(query, (batch_size, offset))

            if not result:
                break

            for row in result:
                try:
                    stored_id, stored_video_id, stored_industry_code, stored_feature_hash = row

                    # 使用增强的相似度计算方法
                    is_duplicate, similarity = self.feature_manager.is_duplicate(
                        feature_hash,
                        stored_feature_hash,
                        self.similarity_threshold
                    )

                    self.logger.debug(
                        f"数据库比对结果: 视频 {video_path} 与 video_id:{stored_video_id} 的相似度为 {similarity:.4f}, 是否重复: {is_duplicate}")

                    if similarity > max_similarity:
                        max_similarity = similarity
                        duplicate_info = json.dumps({"video_id": stored_video_id, "industry_code": stored_industry_code})

                    if is_duplicate:
                        self.logger.info(f"在数据库中找到相似视频，相似度: {similarity:.2%}")
                        return True, duplicate_info, similarity

                except Exception as e:
                    self.logger.error(f"计算相似度时出错: {e}")
                    continue

            offset += batch_size

        # 如果没有找到重复视频
        return False, None, max_similarity

    def save_to_database(self, video_path, feature_hash, file_hash, video_id, industry_code):
        """将视频信息保存到数据库

        Args:
            video_path (str): 视频文件路径
            feature_hash (str): 视频特征哈希值
            file_hash (str): 文件哈希值
            video_id (int): 视频ID
            industry_code (str): 行业代码

        Returns:
            tuple: (bool, int) 保存是否成功及新插入记录的ID
        """
        self.logger.debug("正在将视频信息添加到数据库...")
        cursor = None
        try:
            query = "INSERT INTO videos (video_id, industry_code, feature_hash, file_hash) VALUES (%s, %s, %s, %s)"
            cursor = self.db.execute_query(query, (video_id, industry_code, feature_hash, file_hash))
            if cursor is None:
                self.logger.error("视频信息添加到数据库失败")
                return False, None
            
            # 获取新插入记录的ID
            id = cursor.lastrowid
            self.logger.debug(f"视频信息已成功添加到数据库，ID: {id}")
            return True, id
        except Exception as e:
            self.logger.error(f"添加视频到数据库时出错: {e}")
            return False
        finally:
            if cursor:
                cursor.close()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def update_cache(self, video_id, industry_code, feature_hash, file_hash):
        """更新Redis缓存

        使用tenacity库实现重试机制，包括：
        1. 最多重试3次
        2. 每次重试间隔2秒

        Args:
            video_id (int): 视频ID
            industry_code (str): 行业代码
            feature_hash (str): 视频特征哈希值
            file_hash (str): 文件哈希值

        Returns:
            bool: 更新是否成功
        """
        if not self.redis.cache_enabled:
            return True

        self.logger.debug("更新Redis缓存...")
        try:
            # 使用事务确保缓存更新的原子性
            with self.redis.pipeline() as pipe:
                # 使用JSON格式存储video_id和industry_code
                cache_value = f"{{\"video_id\":{video_id},\"industry_code\":\"{industry_code}\"}}"
                pipe.set(f"file_hash:{file_hash}", cache_value)
                pipe.set(f"feature_hash:{feature_hash}", cache_value)
                # 设置缓存过期时间
                cache_expire = int(os.getenv('CACHE_EXPIRE', '0'))
                if cache_expire > 0:
                    pipe.expire(f"file_hash:{file_hash}", cache_expire)
                    pipe.expire(f"feature_hash:{feature_hash}", cache_expire)
                pipe.execute()
                self.logger.debug(f"Redis缓存更新完成")
                return True
        except Exception as e:
            self.logger.error(f"Redis缓存更新失败: {e}")
            # 清理可能部分更新的缓存
            try:
                self.redis.delete(f"file_hash:{file_hash}")
                self.redis.delete(f"feature_hash:{feature_hash}")
                self.logger.info("已清理部分更新的缓存")
            except Exception as cleanup_error:
                self.logger.error(f"清理缓存失败: {cleanup_error}")
            raise  # 抛出异常以触发重试机制

    def add_video(self, video_path, video_id=None, industry_code=None):
        """将视频添加到数据库并更新缓存

        该方法完成以下任务：
        1. 提取视频特征
        2. 计算文件哈希值
        3. 将视频信息存入数据库
        4. 更新Redis缓存

        Args:
            video_path (str): 待添加视频的文件路径
            video_id (int, optional): 业务系统视频ID
            industry_code (str, optional): 业务系统视频分类代码

        Returns:
            bool: 添加是否成功
        """
        self.logger.info(f"开始添加视频: {video_path}")

        # 检查文件是否存在
        video_path = Path(video_path)
        if not video_path.exists():
            self.logger.error(f"视频文件不存在: {video_path}")
            return False

        # 将Path对象转换为字符串
        video_path_str = str(video_path)

        # 如果未提供video_id，生成一个新的
        if video_id is None:
            video_id = int(time.time())

        # 如果未提供industry_code，使用默认值
        if industry_code is None:
            industry_code = "default"

        self.logger.debug("开始提取视频特征...")
        features = self.feature_manager.extract_video_features(video_path)
        feature_hash = self.feature_manager.calculate_feature_hash(features)
        file_hash = self.feature_manager.calculate_file_hash(video_path)
        self.logger.debug("视频特征提取完成")

        # 使用事务管理器确保数据库操作的原子性
        try:
            # 开启事务
            self.db.connection.start_transaction()

            # 保存到数据库
            db_success, video_id = self.save_to_database(video_path_str, feature_hash, file_hash, video_id, industry_code)
            if not db_success:
                self.logger.error("数据库保存失败，执行回滚")
                self.db.connection.rollback()
                return False

            # 更新缓存，使用视频ID
            try:
                cache_success = self.update_cache(video_id, industry_code, feature_hash, file_hash)
                if not cache_success:
                    self.logger.error("缓存更新失败，执行回滚")
                    self.db.connection.rollback()
                    return False
            except Exception as cache_error:
                self.logger.error(f"缓存更新时发生错误: {str(cache_error)}")
                self.db.connection.rollback()
                return False

            # 提交事务
            self.db.connection.commit()
            self.logger.debug("数据库事务提交成功")

        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            if self.db.connection:
                try:
                    self.db.connection.rollback()
                    self.logger.info("已回滚数据库事务")
                except Exception as rollback_error:
                    self.logger.error(f"回滚失败: {str(rollback_error)}")
            return False

        self.logger.info("视频添加完成")
        return True

    def process_new_video(self, video_path):
        """处理新视频：检查重复并决定是否添加到数据库

        该方法整合了视频查重和添加的完整流程：
        1. 首先检查视频是否重复
        2. 如果不重复，则添加到数据库
        3. 如果重复，则输出重复信息

        Args:
            video_path (str): 待处理视频的文件路径

        Returns:
            bool: 处理是否成功（不重复且添加成功返回True）
        """
        is_duplicate, duplicate_path, similarity = self.check_video(video_path)

        if is_duplicate:
            self.logger.info(f"发现重复视频：")
            self.logger.info(f"新视频: {video_path}")
            self.logger.info(f"已存在视频: {duplicate_path}")
            self.logger.info(f"相似度: {similarity:.2%}")
            return False
        else:
            self.add_video(video_path)
            self.logger.info(f"视频已添加到数据库: {video_path}")
            return True