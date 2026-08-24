import os
import json
import logging
from typing import Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache_service")

# Try to import redis structure cleanly
try:
    import redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    # Lazy client initialization
    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=2.0
    )
    # Check connect status
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Connected to Redis server successfully.")
except Exception as e:
    logger.warning(f"Redis not available or connection failed, falling back to local memory cache: {e}")
    REDIS_AVAILABLE = False
    redis_client = None

# Local high-performance in-memory cache fallback for sandbox/minimal dependencies
class InMemCache:
    def __init__(self):
        self._store = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None):
        self._store[key] = value

    def delete(self, key: str):
        if key in self._store:
            del self._store[key]

_local_cache = InMemCache()

class CacheService:
    @staticmethod
    def get(key: str) -> Optional[Any]:
        try:
            if REDIS_AVAILABLE and redis_client:
                data = redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                data = _local_cache.get(key)
                if data:
                    return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get failure for {key}: {e}")
        return None

    @staticmethod
    def set(key: str, value: Any, expire_seconds: Optional[int] = 300) -> bool:
        try:
            serialized = json.dumps(value)
            if REDIS_AVAILABLE and redis_client:
                return redis_client.set(key, serialized, ex=expire_seconds)
            else:
                _local_cache.set(key, serialized, ex=expire_seconds)
                return True
        except Exception as e:
            logger.error(f"Cache set failure for {key}: {e}")
            return False

    @staticmethod
    def delete(key: str) -> bool:
        try:
            if REDIS_AVAILABLE and redis_client:
                return bool(redis_client.delete(key))
            else:
                _local_cache.delete(key)
                return True
        except Exception as e:
            logger.error(f"Cache delete failure for {key}: {e}")
            return False
