import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis.asyncio as aioredis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based exact match cache for RAG queries."""

    def __init__(self, redis_client: aioredis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = int(timedelta(hours=settings.ttl_hours).total_seconds())
        self.session_ttl = int(timedelta(hours=settings.session_ttl_hours).total_seconds())

    def _generate_cache_key(self, request: AskRequest) -> str:
        """Generate exact cache key based on request parameters."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        # 16 hex chars = 64 bits of the SHA-256 digest. Collision probability
        # is negligible for a bounded query cache (birthday bound ~2^32 entries).
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis GET operation - O(1)
            cached_response = await self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info(f"Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response for exact query matching."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis SET operation with TTL
            success = await self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            if success:
                logger.info(f"Stored response in exact cache with key {cache_key[:16]}...")
                return True
            else:
                logger.warning(f"Failed to store response in cache")
                return False

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    async def get_session_history(self, session_id: str) -> list[dict]:
        """Return up to last 10 messages (5 turns) for the session."""
        key = f"session:{session_id}:history"
        try:
            raw = await self.redis.get(key)
            if not raw:
                return []
            history = json.loads(raw)
            return history[-10:]  # last 5 turns = 10 messages
        except json.JSONDecodeError:
            return []
        except Exception as e:
            logger.error(f"Error fetching session history: {e}")
            return []

    async def append_to_session_history(
        self, session_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        """Append a user/assistant turn atomically and reset the session TTL.

        Uses a Redis pipeline with WATCH to prevent lost updates from concurrent
        requests on the same session. History is trimmed to the last 10 messages
        (5 turns) before storing to prevent unbounded growth.
        """
        key = f"session:{session_id}:history"
        try:
            async with self.redis.pipeline() as pipe:
                while True:
                    try:
                        await pipe.watch(key)
                        raw = await pipe.get(key)
                        history = json.loads(raw) if raw else []
                        history.append({"role": "user", "content": user_msg})
                        history.append({"role": "assistant", "content": assistant_msg})
                        # Keep only the last 10 messages (5 turns)
                        history = history[-10:]
                        pipe.multi()
                        pipe.set(key, json.dumps(history), ex=self.session_ttl)
                        await pipe.execute()
                        break
                    except Exception:
                        # WATCH fired — another request modified the key; retry
                        continue
        except Exception as e:
            logger.error(f"Error appending to session history for {session_id}: {e}")
