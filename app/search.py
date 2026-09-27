"""Document insertion and semantic search execution logic using PostgreSQL and pgvector."""
from typing import List, Dict, Any
import asyncpg
import numpy as np
from app.chunking import chunk_text
from app.embedding import embed, embed_batch


async def insert_document(pool: asyncpg.Pool, title: str, content: str) -> int:
    """
    Chunk text, generate embeddings in batch, and insert each chunk into the documents table.

    Args:
        pool: asyncpg Connection Pool.
        title: Document title (used as parent_title).
        content: Document content.

    Returns:
        int: Number of chunks inserted.
    """
    chunks = chunk_text(content)
    if not chunks:
        return 0

    embeddings = embed_batch(chunks)

    records = [
        (title, i, chunk, np.array(emb, dtype=np.float32))
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    query = """
        INSERT INTO documents (parent_title, chunk_index, content, embedding)
        VALUES ($1, $2, $3, $4);
    """
    async with pool.acquire() as conn:
        await conn.executemany(query, records)

    return len(chunks)


async def search_documents(
    pool: asyncpg.Pool,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Embed query text and search documents using pgvector cosine similarity.

    Args:
        pool: asyncpg Connection Pool.
        query: Query string.
        top_k: Number of most similar results to return (default: 5).

    Returns:
        List[Dict[str, Any]]: Search results with id, parent_title, content, and similarity score.
    """
    query_vector = np.array(embed(query), dtype=np.float32)

    sql = """
        SELECT
            id,
            parent_title,
            content,
            1 - (embedding <=> $1) AS similarity
        FROM documents
        ORDER BY embedding <=> $1
        LIMIT $2;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, query_vector, top_k)

    return [
        {
            "id": row["id"],
            "parent_title": row["parent_title"],
            "content": row["content"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]
