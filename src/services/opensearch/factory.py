from functools import lru_cache

from src.config import get_settings

from .client import OpenSearchClient

@lru_cache(maxsize=1)
def make_opensearch_client() -> OpenSearchClient:
    settings = get_settings()
    return OpenSearchClient(
        host=settings.opensearch.host,
        settings=settings
    )