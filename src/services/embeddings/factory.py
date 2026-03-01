from typing import Optional

from src.config import Settings,get_settings

from .jina_client import JinaEmbeddingsClient

def make_embeddings_service(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key

    if not api_key:
        raise ValueError("Jina API key is not set in settings")

    return JinaEmbeddingsClient(api_key=api_key)

def make_embeddings_client(settings:Optional[Settings]=None) -> JinaEmbeddingsClient:
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key

    if not api_key:
        raise ValueError("Jina API key is not set in settings")

    return JinaEmbeddingsClient(api_key=api_key)