"""Unit tests for embedding generation."""
from app.embedding import embed, embed_batch


def test_embed_dimension():
    """embed() should return a float vector of exactly 384 dimensions."""
    text = "How to perform semantic search using vector databases?"
    vector = embed(text)

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(val, float) for val in vector)


def test_embed_batch():
    """embed_batch() should return a list of 384-dimensional vectors."""
    texts = [
        "What is PostgreSQL pgvector?",
        "FastAPI with asynchronous database connection pool.",
        "Sentence transformers for semantic text representation."
    ]
    vectors = embed_batch(texts, batch_size=2)

    assert len(vectors) == len(texts)
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) == 384
