"""OpenAdapt Retrieval - Multimodal demo retrieval for GUI automation.

This package provides a unified interface for creating multimodal embeddings
from screenshots and task descriptions, enabling semantic demo retrieval
for GUI automation agents.
"""

from openadapt_retrieval.embeddings import (
    BaseEmbedder,
    Qwen3VLEmbedder,
    CLIPEmbedder,
    get_embedder,
)
from openadapt_retrieval.retriever import (
    MultimodalDemoRetriever,
    VectorIndex,
    RetrievalResult,
    DemoMetadata,
)
from openadapt_retrieval.storage import EmbeddingStorage

__version__ = "0.1.0"

__all__ = [
    # Embedders
    "BaseEmbedder",
    "Qwen3VLEmbedder",
    "CLIPEmbedder",
    "get_embedder",
    # Retriever
    "MultimodalDemoRetriever",
    "VectorIndex",
    "RetrievalResult",
    "DemoMetadata",
    # Storage
    "EmbeddingStorage",
]
