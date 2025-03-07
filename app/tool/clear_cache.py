import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dotenv import load_dotenv
from app.db.redis_connection import RedisConnection
from app.services.logger import get_logger

def clear_cache():
    """清空Redis缓存

    清除所有视频相关的缓存数据，包括：
    1. 文件哈希缓存（以'file_hash:'为前缀的键）
    2. 特征哈希缓存（以'feature_hash:'为前缀的键）

    Returns:
        tuple: (success, message)
        - success (bool): 清理是否成功
        - message (str): 操作结果描述
    """
    # 初始化日志记录器
    logger = get_logger()
    logger.info("开始清理Redis缓存...")

    # 加载环境变量
    load_dotenv()

    # 获取Redis连接
    redis = RedisConnection()

    if not redis.cache_enabled:
        logger.info("Redis缓存未启用，无需清理")
        return True, "Redis缓存未启用，无需清理"

    try:
        # 检查Redis连接
        if not redis.connect():
            logger.error("无法连接到Redis服务器")
            return False, "无法连接到Redis服务器"

        # 获取所有缓存键
        file_keys = redis.keys("file_hash:*")
        path_keys = redis.keys("feature_hash:*")

        total_keys = len(file_keys) + len(path_keys)
        logger.info(f"找到 {total_keys} 个缓存键待清理")

        # 使用管道批量删除键
        pipe = redis.pipeline()
        if pipe is None:
            logger.error("无法创建Redis管道")
            return False, "无法创建Redis管道"

        # 删除所有缓存键
        for key in file_keys:
            pipe.delete(key)
        for key in path_keys:
            pipe.delete(key)

        # 执行删除操作
        pipe.execute()

        # 关闭Redis连接
        redis.close()

        success_message = f"成功清理 {total_keys} 个缓存键"
        logger.info(success_message)
        return True, success_message

    except Exception as e:
        error_message = f"清理缓存时发生错误: {str(e)}"
        logger.error(error_message)
        return False, error_message

if __name__ == "__main__":
    success, message = clear_cache()
    print(message)