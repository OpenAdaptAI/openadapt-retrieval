"""Tests for the embeddings module.

These tests verify the embedding interfaces work correctly.
GPU tests are marked with pytest.mark.gpu and require a CUDA device.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a sample image for testing."""
    img = Image.new("RGB", (256, 256), color="red")
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


@pytest.fixture
def sample_text() -> str:
    """Sample text for embedding tests."""
    return "Turn off Night Shift in System Settings"


class TestBaseEmbedder:
    """Tests for BaseEmbedder interface."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1.0."""
        from openadapt_retrieval.embeddings.base import BaseEmbedder

        # Create a mock embedder to test the utility method
        class MockEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self) -> int:
                return 512

            @property
            def model_name(self) -> str:
                return "mock"

            def embed_text(self, text):
                return np.random.randn(512).astype(np.float32)

            def embed_image(self, image):
                return np.random.randn(512).astype(np.float32)

            def embed_multimodal(self, text, image):
                return np.random.randn(512).astype(np.float32)

        embedder = MockEmbedder()
        vec = np.random.randn(512).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize

        similarity = embedder.cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        from openadapt_retrieval.embeddings.base import BaseEmbedder

        class MockEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self) -> int:
                return 2

            @property
            def model_name(self) -> str:
                return "mock"

            def embed_text(self, text):
                return np.array([1, 0], dtype=np.float32)

            def embed_image(self, image):
                return np.array([0, 1], dtype=np.float32)

            def embed_multimodal(self, text, image):
                return np.array([1, 1], dtype=np.float32)

        embedder = MockEmbedder()
        vec1 = np.array([1, 0], dtype=np.float32)
        vec2 = np.array([0, 1], dtype=np.float32)

        similarity = embedder.cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001


class TestRegistry:
    """Tests for embedder registry."""

    def test_list_embedders(self):
        """Test listing available embedders."""
        from openadapt_retrieval.embeddings.registry import list_embedders

        embedders = list_embedders()
        assert "qwen3vl" in embedders
        assert "clip" in embedders
        assert "qwen" in embedders  # Alias

    def test_get_embedder_invalid(self):
        """Test that invalid embedder name raises ValueError."""
        from openadapt_retrieval.embeddings.registry import get_embedder

        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("nonexistent_embedder")


class TestQwen3VLEmbedderConfig:
    """Tests for Qwen3VLEmbedder configuration."""

    def test_config_validation_dim_too_large(self):
        """Test that embedding_dim > max raises error."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        with pytest.raises(ValueError, match="exceeds model maximum"):
            Qwen3VLEmbedder(embedding_dim=10000)

    def test_config_validation_dim_too_small(self):
        """Test that embedding_dim < 64 raises error."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        with pytest.raises(ValueError, match="at least 64"):
            Qwen3VLEmbedder(embedding_dim=32)

    def test_config_default_dim(self):
        """Test default embedding dimension."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        embedder = Qwen3VLEmbedder(embedding_dim=512)
        assert embedder.embedding_dim == 512


class TestMockEmbedding:
    """Tests using mock models (no GPU required)."""

    def test_embed_batch_with_mock(self, sample_image: Path, sample_text: str):
        """Test batch embedding with mocked model."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        embedder = Qwen3VLEmbedder(embedding_dim=512)

        # Mock the _embed method to return random embeddings
        def mock_embed(text=None, image=None, instruction=None):
            return np.random.randn(512).astype(np.float32)

        with patch.object(embedder, "_embed", mock_embed):
            with patch.object(embedder, "_load_model", lambda: None):
                inputs = [
                    {"text": sample_text},
                    {"text": "Another task", "image": str(sample_image)},
                    {"image": str(sample_image)},
                ]

                embeddings = embedder.embed_batch(inputs, show_progress=False)

                assert embeddings.shape == (3, 512)
                assert embeddings.dtype == np.float32


@pytest.mark.gpu
@pytest.mark.slow
class TestQwen3VLEmbedderGPU:
    """GPU tests for Qwen3VLEmbedder.

    These tests require a CUDA device and download the model.
    Run with: pytest -m gpu
    """

    def test_embed_text_only(self, sample_text: str):
        """Test text-only embedding."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        embedder = Qwen3VLEmbedder(embedding_dim=512)
        embedding = embedder.embed_text(sample_text)

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        # Normalized vectors should have norm ~1.0
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01

    def test_embed_image_only(self, sample_image: Path):
        """Test image-only embedding."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        embedder = Qwen3VLEmbedder(embedding_dim=512)
        embedding = embedder.embed_image(sample_image)

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32

    def test_embed_multimodal(self, sample_image: Path, sample_text: str):
        """Test text+image embedding."""
        from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder

        embedder = Qwen3VLEmbedder(embedding_dim=512)
        embedding = embedder.embed_multimodal(text=sample_text, image=sample_image)

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32


class TestCLIPEmbedder:
    """Tests for CLIP embedder."""

    def test_clip_model_name(self):
        """Test CLIP model name property."""
        from openadapt_retrieval.embeddings.clip import CLIPEmbedder

        embedder = CLIPEmbedder()
        assert "clip" in embedder.model_name.lower()
        assert "ViT-L-14" in embedder.model_name

    @pytest.mark.skipif(
        True,  # Skip by default - requires open_clip
        reason="Requires open-clip-torch package"
    )
    def test_clip_embed_text(self, sample_text: str):
        """Test CLIP text embedding."""
        from openadapt_retrieval.embeddings.clip import CLIPEmbedder

        embedder = CLIPEmbedder()
        embedding = embedder.embed_text(sample_text)

        assert embedding.shape == (embedder.embedding_dim,)
        assert embedding.dtype == np.float32
