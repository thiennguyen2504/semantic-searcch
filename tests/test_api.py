"""Tests for API endpoints (/health, /documents, /search)."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


def test_post_documents_validation():
    with TestClient(app) as client:
        # Empty content should return 400
        res = client.post("/documents", json={"title": "Test Title", "content": ""})
        assert res.status_code == 400

        # Empty title should return 400
        res = client.post("/documents", json={"title": "", "content": "Sample content"})
        assert res.status_code == 400


def test_search_validation():
    with TestClient(app) as client:
        # Empty query should return 400
        res = client.get("/search?q=&top_k=5")
        assert res.status_code == 400

        # top_k < 1 should return 400
        res = client.get("/search?q=test&top_k=0")
        assert res.status_code == 400

        # top_k > 50 should return 400
        res = client.get("/search?q=test&top_k=51")
        assert res.status_code == 400


def test_insert_and_search_flow():
    with TestClient(app) as client:
        # Insert a document
        doc_data = {
            "title": "Java NullPointerException handling",
            "content": "To prevent NullPointerException in Java, check for null before invoking methods or use Optional."
        }
        res_insert = client.post("/documents", json=doc_data)
        assert res_insert.status_code == 201
        data_out = res_insert.json()
        assert "document_id" in data_out
        assert data_out["num_chunks"] >= 1

        # Search for it
        res_search = client.get("/search?q=how to avoid null pointer in java&top_k=3")
        assert res_search.status_code == 200
        results = res_search.json()
        assert isinstance(results, list)
        assert len(results) > 0
        first = results[0]
        assert "id" in first
        assert "parent_title" in first
        assert "content" in first
        assert "similarity" in first
        assert isinstance(first["similarity"], float)
