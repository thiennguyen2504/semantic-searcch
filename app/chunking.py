"""Text chunking utilities using tiktoken."""
from typing import List
import tiktoken

# Use cl100k_base tokenizer
_tokenizer = tiktoken.get_encoding("cl100k_base")


def get_tokenizer():
    """Return the global tiktoken tokenizer instance."""
    return _tokenizer


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of tokens using tiktoken.

    Args:
        text (str): Input text string to be chunked.
        chunk_size (int): Maximum number of tokens per chunk (default: 300).
        overlap (int): Number of overlapping tokens between consecutive chunks (default: 50).

    Returns:
        List[str]: List of text chunks. If text token length <= chunk_size, returns [text].
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be strictly positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    if not text:
        return []

    tokens = _tokenizer.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    stride = chunk_size - overlap

    chunks: List[str] = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunks.append(_tokenizer.decode(chunk_tokens))
        if end == total_tokens:
            break
        start += stride

    return chunks
