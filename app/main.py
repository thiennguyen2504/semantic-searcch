import logging
import uuid
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Query, status
from app.database import init_db, close_db, get_pool
from app.schemas import DocumentIn, DocumentOut, SearchResult
from app.search import insert_document, search_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("semantic_search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB extension, table, and connection pool
    logger.info("Application starting up: initializing database...")
    await init_db()
    yield
    # Shutdown: Close database pool
    logger.info("Application shutting down: closing database connection...")
    await close_db()


app = FastAPI(
    title="Semantic Search API",
    description="Vector search API using FastAPI, PostgreSQL and pgvector",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def create_document(doc: DocumentIn):
    """
    Insert a document by chunking its content, embedding chunks, and storing in DB.
    """
    if not doc.content or not doc.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document content cannot be empty."
        )
    if not doc.title or not doc.title.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document title cannot be empty."
        )

    pool = get_pool()
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection pool is not initialized."
        )

    num_chunks = await insert_document(pool, doc.title.strip(), doc.content.strip())
    document_id = str(uuid.uuid4())

    return DocumentOut(
        document_id=document_id,
        num_chunks=num_chunks,
    )


@app.get(
    "/search",
    response_model=List[SearchResult],
    tags=["Search"],
)
async def search(
    q: str = Query(..., description="Query text to search for"),
    top_k: int = Query(default=5, description="Number of results to retrieve (1-50)"),
):
    """
    Semantic search across indexed document chunks using cosine similarity.
    """
    if not q or not q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query 'q' cannot be empty."
        )
    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be between 1 and 50."
        )

    pool = get_pool()
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection pool is not initialized."
        )

    results = await search_documents(pool, q.strip(), top_k=top_k)
    return results
