from typing import Optional

from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: Optional[str] = None

class TextChunk(BaseModel):
    text: str
    metadata: ChunkMetadata
    arxiv_id :str
    paper_id :str