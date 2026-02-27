from typing import Optional

from src.config import Settings,get_settings

from .jina_client import JinaEnbeddingsClient

def make_embeddings_service(settings: Optional[Settings] = None) -> JinaEnbeddingsClient:
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key

    if not api_key:
        raise ValueError("Jina API key is not set in settings")

    return JinaEnbeddingsClient(api_key=api_key)

def make_embeddings_client(settings:Optional[Settings]=None) -> JinaEnbeddingsClient:
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key

    if not api_key:
        raise ValueError("Jina API key is not set in settings")

    return JinaEnbeddingsClient(api_key=api_key)