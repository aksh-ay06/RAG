from typing import Dict, List

from pydantic import BaseModel

class JinaEmbeddingRequest(BaseModel):
    model: str = "jina-embeddings-v3"
    task: str = "retrival.passage"
    dimensions: int = 1024
    late_chunking: bool = False
    embedding_type: str = "float"
    input: List[str]

class JinaEmbeddingResponse(BaseModel):
    model: str
    object: str
    usage: Dict[str, int]
    data: List[Dict]