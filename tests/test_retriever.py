"""Tests for the retriever module.

These tests verify the demo retrieval pipeline works correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a sample image for testing."""
    img = Image.new("RGB", (256, 256), color="blue")
    path = tmp_path / "test_screenshot.png"
    img.save(path)
    return path


@pytest.fixture
def sample_demos() -> list[dict]:
    """Sample demo data for testing."""
    return [
        {
            "demo_id": "turn-off-nightshift",
            "task": "Turn off Night Shift in System Settings",
            "app_name": "System Settings",
            "platform": "macos",
        },
        {
            "demo_id": "search-github",
            "task": "Search for Python projects on GitHub",
            "domain": "github.com",
            "platform": "web",
        },
        {
            "demo_id": "open-calculator",
            "task": "Open the Calculator application",
            "app_name": "Finder",
            "platform": "macos",
        },
    ]


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""
    embedder = Mock()
    embedder.embedding_dim = 512
    embedder.model_name = "mock-embedder"

    def embed_text(text):
        # Create deterministic embedding based on text hash
        np.random.seed(abs(hash(text)) % (2**31))
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # Normalize
        return emb.astype(np.float32)

    def embed_image(image):
        np.random.seed(42)
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # Normalize
        return emb.astype(np.float32)

    def embed_multimodal(text, image):
        text_emb = embed_text(text)
        img_emb = embed_image(image)
        combined = (text_emb + img_emb) / 2
        combined = combined / np.linalg.norm(combined)
        return combined.astype(np.float32)

    def embed_batch(inputs, show_progress=True):
        embeddings = []
        for inp in inputs:
            text = inp.get("text")
            image = inp.get("image")
            if text and image:
                emb = embed_multimodal(text, image)
            elif text:
                emb = embed_text(text)
            else:
                emb = embed_image(image)
            embeddings.append(emb)
        return np.stack(embeddings).astype(np.float32)

    embedder.embed_text = embed_text
    embedder.embed_image = embed_image
    embedder.embed_multimodal = embed_multimodal
    embedder.embed_batch = embed_batch

    return embedder


class TestDemoMetadata:
    """Tests for DemoMetadata class."""

    def test_create_demo_metadata(self):
        """Test creating demo metadata."""
        from openadapt_retrieval.retriever import DemoMetadata

        demo = DemoMetadata(
            demo_id="test-demo",
            task="Test task description",
            app_name="Test App",
        )

        assert demo.demo_id == "test-demo"
        assert demo.task == "Test task description"
        assert demo.app_name == "Test App"
        assert demo.tags == []
        assert demo.metadata == {}

    def test_demo_metadata_serialization(self):
        """Test demo metadata serialization."""
        from openadapt_retrieval.retriever import DemoMetadata

        demo = DemoMetadata(
            demo_id="test-demo",
            task="Test task",
            tags=["settings", "display"],
            metadata={"priority": 1},
        )

        data = demo.model_dump()
        assert data["demo_id"] == "test-demo"
        assert data["tags"] == ["settings", "display"]
        assert data["metadata"]["priority"] == 1


class TestVectorIndex:
    """Tests for VectorIndex class."""

    def test_create_index(self):
        """Test creating a vector index."""
        from openadapt_retrieval.retriever import VectorIndex

        index = VectorIndex(dimension=512)
        assert index.dimension == 512
        assert index.size == 0

    def test_add_and_search(self):
        """Test adding vectors and searching."""
        from openadapt_retrieval.retriever import VectorIndex

        index = VectorIndex(dimension=4, use_faiss=False)  # Use numpy for testing

        # Add some vectors
        vectors = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        index.add(vectors)

        assert index.size == 4

        # Search for vector similar to first one
        query = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
        scores, indices = index.search(query, top_k=2)

        assert len(scores) == 2
        assert len(indices) == 2
        assert indices[0] == 0  # Most similar to first vector

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading index."""
        from openadapt_retrieval.retriever import VectorIndex

        index = VectorIndex(dimension=4, use_faiss=False)
        vectors = np.random.randn(10, 4).astype(np.float32)
        index.add(vectors)

        # Save
        index.save(tmp_path / "test_index")

        # Load in new index
        index2 = VectorIndex(dimension=4, use_faiss=False)
        index2.load(tmp_path / "test_index")

        assert index2.size == 10

    def test_empty_index_search_raises(self):
        """Test that searching empty index raises error."""
        from openadapt_retrieval.retriever import VectorIndex

        index = VectorIndex(dimension=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)

        with pytest.raises(ValueError, match="Index is empty"):
            index.search(query, top_k=1)


class TestMultimodalDemoRetriever:
    """Tests for MultimodalDemoRetriever."""

    def test_add_demo(self, sample_demos):
        """Test adding demos to retriever."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()

        for demo_data in sample_demos:
            retriever.add_demo(**demo_data)

        assert len(retriever) == 3
        assert retriever.get_demo_count() == 3

    def test_get_demo(self, sample_demos):
        """Test getting demo by ID."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()
        retriever.add_demos(sample_demos)

        demo = retriever.get_demo("turn-off-nightshift")
        assert demo is not None
        assert demo.task == "Turn off Night Shift in System Settings"

        missing = retriever.get_demo("nonexistent")
        assert missing is None

    def test_build_index_with_mock_embedder(self, sample_demos, mock_embedder):
        """Test building index with mock embedder."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()
        retriever._embedder = mock_embedder
        retriever.config.use_faiss = False  # Use numpy for testing

        retriever.add_demos(sample_demos)
        retriever.build_index()

        assert retriever._is_indexed
        assert retriever._embeddings is not None
        assert retriever._embeddings.shape == (3, 512)

    def test_build_index_empty_raises(self):
        """Test that building index with no demos raises error."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()

        with pytest.raises(ValueError, match="no demos added"):
            retriever.build_index()

    def test_retrieve_with_mock_embedder(self, sample_demos, mock_embedder):
        """Test retrieval with mock embedder."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()
        retriever._embedder = mock_embedder
        retriever.config.use_faiss = False  # Use numpy for testing

        retriever.add_demos(sample_demos)
        retriever.build_index()

        results = retriever.retrieve(
            task="Disable Night Shift",
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2
        # Scores can be negative for random embeddings (cosine similarity ranges from -1 to 1)
        # The important thing is that results are ranked by score (highest first)
        assert results[0].score >= results[1].score

    def test_retrieve_not_indexed_raises(self, sample_demos):
        """Test that retrieval without indexing raises error."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()
        retriever.add_demos(sample_demos)

        with pytest.raises(ValueError, match="Index not built"):
            retriever.retrieve(task="test query")

    def test_context_bonus(self, sample_demos, mock_embedder):
        """Test that app/domain context provides score bonus."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()
        retriever._embedder = mock_embedder
        retriever.config.use_faiss = False  # Use numpy for testing

        retriever.add_demos(sample_demos)
        retriever.build_index()

        # Search without context
        results_no_context = retriever.retrieve(task="test", top_k=3)

        # Search with matching context
        results_with_context = retriever.retrieve(
            task="test",
            top_k=3,
            app_context="System Settings",
        )

        # The demo with matching app should get a bonus
        nightshift_no_ctx = next(
            r for r in results_no_context if r.demo.demo_id == "turn-off-nightshift"
        )
        nightshift_with_ctx = next(
            r for r in results_with_context if r.demo.demo_id == "turn-off-nightshift"
        )

        assert nightshift_with_ctx.score > nightshift_no_ctx.score

    def test_save_and_load(self, sample_demos, mock_embedder, tmp_path: Path):
        """Test saving and loading retriever state."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        # Create and build retriever
        retriever = MultimodalDemoRetriever(index_path=tmp_path / "test_index")
        retriever._embedder = mock_embedder
        retriever.config.use_faiss = False  # Use numpy for testing

        retriever.add_demos(sample_demos)
        retriever.build_index()
        retriever.save()

        # Load in new retriever
        retriever2 = MultimodalDemoRetriever(index_path=tmp_path / "test_index")
        retriever2._embedder = mock_embedder
        retriever2.config.use_faiss = False  # Use numpy for testing
        retriever2.load()

        assert len(retriever2) == 3
        assert retriever2._is_indexed

        # Verify retrieval works
        results = retriever2.retrieve(task="Night Shift", top_k=1)
        assert len(results) == 1

    def test_clear(self, sample_demos, mock_embedder):
        """Test clearing retriever."""
        from openadapt_retrieval.retriever import MultimodalDemoRetriever

        retriever = MultimodalDemoRetriever()
        retriever._embedder = mock_embedder
        retriever.config.use_faiss = False  # Use numpy for testing

        retriever.add_demos(sample_demos)
        retriever.build_index()

        assert len(retriever) == 3

        retriever.clear()

        assert len(retriever) == 0
        assert not retriever._is_indexed


class TestEmbeddingStorage:
    """Tests for EmbeddingStorage."""

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading embeddings."""
        from openadapt_retrieval.storage import EmbeddingStorage

        storage = EmbeddingStorage(tmp_path / "test_storage")

        embeddings = np.random.randn(5, 512).astype(np.float32)
        metadata = [{"id": f"demo_{i}"} for i in range(5)]
        config = {"model": "test", "dim": 512}

        storage.save(embeddings, metadata, config)

        assert storage.exists()

        # Load
        data = storage.load()
        assert data["embeddings"].shape == (5, 512)
        assert len(data["metadata"]) == 5
        assert data["config"]["model"] == "test"

    def test_get_info(self, tmp_path: Path):
        """Test getting storage info."""
        from openadapt_retrieval.storage import EmbeddingStorage

        storage = EmbeddingStorage(tmp_path / "test_storage")

        # Non-existent
        info = storage.get_info()
        assert not info["exists"]

        # After saving
        embeddings = np.random.randn(10, 256).astype(np.float32)
        storage.save(embeddings)

        info = storage.get_info()
        assert info["exists"]
        assert info["embedding_count"] == 10
        assert info["embedding_dim"] == 256

    def test_append(self, tmp_path: Path):
        """Test appending embeddings."""
        from openadapt_retrieval.storage import EmbeddingStorage

        storage = EmbeddingStorage(tmp_path / "test_storage")

        # Initial save
        embeddings1 = np.random.randn(3, 128).astype(np.float32)
        storage.save(embeddings1)

        # Append
        embeddings2 = np.random.randn(2, 128).astype(np.float32)
        storage.append(embeddings2)

        # Verify
        data = storage.load()
        assert data["embeddings"].shape == (5, 128)

    def test_delete(self, tmp_path: Path):
        """Test deleting storage."""
        from openadapt_retrieval.storage import EmbeddingStorage

        storage = EmbeddingStorage(tmp_path / "test_storage")

        embeddings = np.random.randn(3, 64).astype(np.float32)
        storage.save(embeddings)

        assert storage.exists()

        storage.delete()

        assert not storage.exists()
