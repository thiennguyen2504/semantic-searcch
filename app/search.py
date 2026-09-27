"""Semantic search execution logic using PostgreSQL and pgvector."""
from typing import List, Dict, Any
import asyncpg


async def search_similar_documents(
    conn: asyncpg.Connection,
    query_vector: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search documents using cosine distance with pgvector (<=> operator).
    Cosine similarity = 1 - cosine_distance.
    """
    query = """
        SELECT
            id,
            parent_title,
            chunk_index,
            content,
            1 - (embedding <=> $1) AS similarity_score
        FROM documents
        ORDER BY embedding <=> $1
        LIMIT $2;
    """
    # Note: query_vector is converted to pgvector format or passed as list depending on register_vector
    rows = await conn.fetch(query, query_vector, top_k)
    return [dict(row) for row in rows]
