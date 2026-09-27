"""Pydantic schemas for request and response validation."""
from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class DocumentCreate(BaseModel):
    parent_title: str
    chunk_index: int = 0
    content: str


class DocumentResponse(BaseModel):
    id: int
    parent_title: Optional[str] = None
    chunk_index: Optional[int] = None
    content: str
    similarity_score: Optional[float] = None


class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: List[DocumentResponse]
