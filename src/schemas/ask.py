from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="Question to ask about arXiv papers")


class PaperSource(BaseModel):
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of paper authors")
    abstract_preview: str = Field(..., description="Preview of paper abstract")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question")
    sources: List[PaperSource] = Field(..., description="Source papers used for the answer")