"""Pydantic schemas for request and response validation."""
from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class DocumentIn(BaseModel):
    title: str = Field(..., description="Title of the parent document")
    content: str = Field(..., description="Full content of the document")



class DocumentOut(BaseModel):
    document_id: str = Field(..., description="Identifier for the inserted document")
    num_chunks: int = Field(..., description="Number of chunks created and stored")


class SearchResult(BaseModel):
    id: int
    parent_title: str
    content: str
    similarity: float


# Legacy / alternative aliases for flexibility
class DocumentCreate(DocumentIn):
    pass


class DocumentResponse(SearchResult):
    pass


class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
