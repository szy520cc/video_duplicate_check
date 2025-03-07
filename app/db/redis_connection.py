import os
from typing import Optional, Any
import redis
from dotenv import load_dotenv
from app.services.logger import get_logger

# 加载环境变量
load_dotenv()

class RedisConnection:
    """Redis连接管理类
    
    实现了单例模式的Redis连接管理器，用于处理与Redis服务器的连接和数据操作。
    该类提供了基本的Redis操作方法，包括设置/获取键值对、删除键、检查键是否存在等功能。
    所有的键都会自动添加配置的前缀，便于管理和区分不同应用的数据。
    
    Attributes:
        _instance: 单例模式的实例对象
        _initialized: 标记是否已经初始化
        redis_config: Redis连接配置信息
        prefix: Redis键名前缀
        _redis_client: Redis客户端实例
        cache_enabled: 是否启用缓存功能
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        """创建单例实例

        Returns:
            RedisConnection: 返回RedisConnection的单例实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化Redis连接配置
        
        从环境变量中读取Redis配置信息，包括主机地址、端口、密码等。
        只在第一次初始化时执行配置加载。
        """
        if self._initialized:
            return

        # 初始化日志记录器
        self.logger = get_logger()
        self.logger.info("初始化redis连接管理器...")

        # 连接池配置
        pool_config = {
            'host': os.getenv('REDIS_HOST'),
            'port': int(os.getenv('REDIS_PORT')),
            'password': os.getenv('REDIS_PASSWORD'),
            'db': int(os.getenv('REDIS_DB', 0)),
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', 10)),
            'decode_responses': True
        }

        self.cache_enabled = os.getenv('REDIS_CACHE_ENABLED', 'true').lower() == 'true'
        if self.cache_enabled:
            try:
                self._pool = redis.ConnectionPool(**pool_config)
                self.logger.info("Redis连接池初始化成功")
            except redis.RedisError as e:
                self.logger.error(f"Redis连接池初始化失败: {e}")
                self._pool = None
        else:
            self._pool = None

        self._redis_client = None
        self._initialized = True

    @property
    def redis_client(self) -> redis.Redis:
        """获取Redis客户端实例
        
        如果缓存功能未启用，返回None；
        如果客户端未初始化，从连接池创建新的客户端实例。
        
        Returns:
            redis.Redis: Redis客户端实例，如果缓存未启用则返回None
        """
        if not self.cache_enabled or self._pool is None:
            return None
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis(connection_pool=self._pool)
            except redis.RedisError as e:
                self.logger.error(f"从连接池获取Redis连接失败: {e}")
                return None
        return self._redis_client

    def close(self):
        """关闭Redis连接
        
        安全地关闭Redis连接，确保资源被正确释放
        """
        if self._redis_client:
            try:
                self._redis_client.close()
            except redis.RedisError as e:
                self.logger.error(f"Redis关闭连接失败: {e}")
            finally:
                self._redis_client = None

        # 在应用程序退出时释放连接池
        if self._pool:
            try:
                self._pool.disconnect()
                self.logger.info("Redis连接池已释放")
            except redis.RedisError as e:
                self.logger.error(f"Redis连接池释放失败: {e}")
            finally:
                self._pool = None

    def pipeline(self):
        """获取Redis管道对象，用于执行原子性的批量操作
        
        如果缓存功能未启用，返回None；
        否则返回Redis的管道对象。
        
        Returns:
            redis.client.Pipeline: Redis管道对象，如果缓存未启用则返回None
        """
        if not self.cache_enabled:
            return None
        try:
            return self.redis_client.pipeline()
        except redis.RedisError as e:
            self.logger.error(f"获取Redis管道对象失败: {e}")
            return None

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """设置键值对
        
        Args:
            key: 键名
            value: 键值
            ex: 过期时间（秒），可选
            
        Returns:
            bool: 操作是否成功
        """
        if not self.cache_enabled:
            return True
        try:
            return self.redis_client.set(key, value, ex=ex)
        except redis.RedisError as e:
            self.logger.error(f"Redis设置键值对失败: {e}")
            return False

    def get(self, key: str) -> Optional[str]:
        """获取键值
        
        Args:
            key: 键名
            
        Returns:
            Optional[str]: 键对应的值，如果键不存在或发生错误则返回None
        """
        if not self.cache_enabled:
            return None
        try:
            return self.redis_client.get(key)
        except redis.RedisError as e:
            self.logger.error(f"Redis获取键值失败: {e}")
            return None

    def delete(self, key: str) -> bool:
        """删除键
        
        Args:
            key: 要删除的键名
            
        Returns:
            bool: 操作是否成功
        """
        if not self.cache_enabled:
            return True
        try:
            return bool(self.redis_client.delete(key))
        except redis.RedisError as e:
            self.logger.error(f"Redis删除键失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在
        
        Args:
            key: 要检查的键名
            
        Returns:
            bool: 键是否存在
        """
        try:
            return bool(self.redis_client.exists(key))
        except redis.RedisError as e:
            self.logger.error(f"Redis检查键是否存在失败: {e}")
            return False
            
    def keys(self, pattern: str) -> list:
        """获取匹配模式的所有键
        
        Args:
            pattern: 键名匹配模式
            
        Returns:
            list: 匹配的键列表
        """
        if not self.cache_enabled:
            return []
        try:
            keys = self.redis_client.keys(pattern)
            return keys
        except redis.RedisError as e:
            self.logger.error(f"Redis获取键列表失败: {e}")
            return []

    def connect(self) -> bool:
        """尝试建立Redis连接
        
        Returns:
            bool: 连接是否成功
        """
        if not self.cache_enabled:
            return True
            
        try:
            # 尝试ping来测试连接
            return bool(self.redis_client.ping())
        except redis.RedisError as e:
            self.logger.error(f"Redis连接失败: {e}")
            return False

    def close(self):
        """关闭Redis连接
        
        安全地关闭Redis连接，确保资源被正确释放
        """
        if self._redis_client:
            try:
                self._redis_client.close()
            except redis.RedisError as e:
                self.logger.error(f"Redis关闭连接失败: {e}")
            finally:
                self._redis_client = None
