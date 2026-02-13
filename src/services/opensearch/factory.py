from functools import lru_cache

from src.config import get_settings

from .client import OpensearchClient

@lru_cache(maxsize=1)
def make_opensearch_client() -> OpensearchClient:
    settings = get_settings()
    return OpensearchClient(
        host=settings.opensearch.host,
        settings=settings
    )