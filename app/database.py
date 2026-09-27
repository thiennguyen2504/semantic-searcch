import os
import logging
from typing import AsyncGenerator, Optional
import asyncpg
from pgvector.asyncpg import register_vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/semantic_search",
)

logger = logging.getLogger("semantic_search.database")

pool: Optional[asyncpg.Pool] = None


async def init_connection(conn: asyncpg.Connection) -> None:
    """Initialize each connection in the pool with pgvector support."""
    await register_vector(conn)


async def init_db() -> None:
    """Create pgvector extension, documents table, and initialize connection pool."""
    global pool

    logger.info("Connecting to PostgreSQL to initialize database...")
    # Connect directly first to create extension and table before registering vector types
    temp_conn = await asyncpg.connect(DATABASE_URL)
    try:
        await temp_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await temp_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                parent_title TEXT,
                chunk_index INT,
                content TEXT,
                embedding VECTOR(384)
            );
            """
        )
        logger.info("Successfully ensured 'vector' extension and 'documents' table exist.")
    finally:
        await temp_conn.close()

    # Create the connection pool with vector type registration
    if pool is None:
        pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            init=init_connection,
            min_size=2,
            max_size=10,
        )
        logger.info("asyncpg connection pool created successfully.")


async def close_db() -> None:
    """Close connection pool gracefully."""
    global pool
    if pool is not None:
        await pool.close()
        pool = None
        logger.info("asyncpg connection pool closed.")


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """Dependency that yields a database connection from the pool."""
    global pool
    if pool is None:
        raise RuntimeError("Database pool has not been initialized. Call init_db() first.")
    async with pool.acquire() as connection:
        yield connection
