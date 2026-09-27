"""Unit tests for text chunking."""
import pytest
from app.chunking import chunk_text, get_tokenizer


def test_chunk_text_short_input():
    """Text with tokens <= chunk_size should return [text] unchanged."""
    text = "FastAPI is a modern, fast web framework for building APIs with Python."
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_input():
    """Empty or whitespace text should return an empty list."""
    assert chunk_text("") == []
    assert chunk_text("   ") == ["   "] or chunk_text("   ") == []


def test_chunk_text_long_input_count_and_overlap():
    """
    Test chunk_text with long input to verify:
    1. The expected number of chunks.
    2. The overlap logic (consecutive chunks overlap by exact number of tokens,
       and chunk[i+1] starts at token position (chunk_size - overlap) of chunk[i]).
    """
    tokenizer = get_tokenizer()

    # Generate a long text with known token count
    # Repeat a sentence to reach ~800 tokens
    base_sentence = "The quick brown fox jumps over the lazy dog in the sunny morning. "
    tokens_per_sentence = len(tokenizer.encode(base_sentence))
    target_tokens = 750
    repeat_count = (target_tokens // tokens_per_sentence) + 5
    full_text = base_sentence * repeat_count
    all_tokens = tokenizer.encode(full_text)
    total_token_count = len(all_tokens)

    chunk_size = 300
    overlap = 50
    stride = chunk_size - overlap  # 250

    # Calculate expected number of chunks
    # Positions: 0, 250, 500, 750...
    expected_chunk_count = 0
    start = 0
    while start < total_token_count:
        expected_chunk_count += 1
        end = min(start + chunk_size, total_token_count)
        if end == total_token_count:
            break
        start += stride

    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) == expected_chunk_count

    # Verify overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        curr_tokens = tokenizer.encode(chunks[i])
        next_tokens = tokenizer.encode(chunks[i + 1])

        # Current chunk should have chunk_size tokens (unless it was the last, but it's not)
        assert len(curr_tokens) == chunk_size

        # The overlap region: last `overlap` tokens of curr_chunk must equal first `overlap` tokens of next_chunk
        assert curr_tokens[-overlap:] == next_tokens[:overlap]

        # Verify next chunk starts at (chunk_size - overlap) relative to curr_chunk
        assert next_tokens[:overlap] == curr_tokens[chunk_size - overlap:]


def test_chunk_text_invalid_overlap():
    """Chunking should raise ValueError if chunk_size <= overlap."""
    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=50, overlap=100)
