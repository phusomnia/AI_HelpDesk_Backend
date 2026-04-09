import logging
from typing import Optional, Dict
from redis import Redis, ConnectionPool
from redisvl.index import SearchIndex
from langchain_community.storage.redis import RedisStore

log = logging.getLogger(__name__)

class RedisConnectionManager:
    _instance: Optional['RedisConnectionManager'] = None
    _pools: Dict[str, ConnectionPool] = {}
    _indexes: Dict[str, SearchIndex] = {}
    _stores: Dict[str, RedisStore] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pool(self, redis_url: str) -> ConnectionPool:
        """Get or create connection pool"""
        if redis_url not in self._pools:
            log.info(f"[Redis] Creating new connection pool for {redis_url}")
            self._pools[redis_url] = ConnectionPool.from_url(
                redis_url,
                max_connections=50,
                socket_keepalive=True,
                socket_keepalive_options={},
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
        return self._pools[redis_url]
    
    def get_redis(self, redis_url: str) -> Redis:
        """Get Redis client from pool"""
        pool = self.get_pool(redis_url)
        return Redis(connection_pool=pool)
    
    def get_search_index(self, redis_url: str, yaml_path: str = "config_env/redis_index.yaml") -> SearchIndex:
        """Get cached SearchIndex, load YAML once"""
        cache_key = f"{redis_url}:{yaml_path}"
        if cache_key not in self._indexes:
            log.info(f"[Redis] Loading SearchIndex from {yaml_path}")
            index = SearchIndex.from_yaml(yaml_path)
            index.connect(redis_url=redis_url)
            self._indexes[cache_key] = index
        return self._indexes[cache_key]
    
    def get_store(self, redis_url: str) -> RedisStore:
        """Get cached RedisStore"""
        if redis_url not in self._stores:
            log.info(f"[Redis] Creating Redis store for {redis_url}")
            self._stores[redis_url] = RedisStore(redis_url=redis_url)
        return self._stores[redis_url]
    
    def close_all(self):
        """Close all connections"""
        for pool in self._pools.values():
            pool.disconnect()
        self._pools.clear()
        self._indexes.clear()
        self._stores.clear()
        log.info("[Redis] All connections closed")

def get_redis_manager() -> RedisConnectionManager:
    return RedisConnectionManager()
