"""Text chunking utilities."""
from typing import List


def chunk_text_by_length(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple text chunking based on character length with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += chunk_size - overlap
        
    return chunks
