"""Text embedding generation service using sentence-transformers."""
import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Load model once at module level
model = SentenceTransformer(MODEL_NAME)


def embed(text: str) -> List[float]:
    """
    Generate 384-dimensional vector embedding for a single text.

    Args:
        text (str): Input text string.

    Returns:
        List[float]: 384-dimensional embedding vector.
    """
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts (List[str]): List of input text strings.
        batch_size (int): Batch size for inference (default: 64).

    Returns:
        List[List[float]]: List of 384-dimensional embedding vectors.
    """
    if not texts:
        return []
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [emb.tolist() for emb in embeddings]
