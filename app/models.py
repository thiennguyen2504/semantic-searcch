"""Database and domain models for Semantic Search."""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DocumentRecord:
    id: Optional[int] = None
    parent_title: Optional[str] = None
    chunk_index: Optional[int] = None
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
