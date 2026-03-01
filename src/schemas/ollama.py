from typing import List, Optional

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Structured response model for RAG answers from Ollama."""

    answer: str = Field(..., description="The generated answer to the user's question")
    sources: List[str] = Field(default_factory=list, description="List of source PDF URLs")
    confidence: Optional[str] = Field("medium", description="Confidence level: low, medium, high")
    citations: List[str] = Field(default_factory=list, description="List of cited arXiv IDs")