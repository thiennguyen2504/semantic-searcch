import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import init_db, close_db

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
