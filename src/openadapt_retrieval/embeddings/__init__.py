"""Embeddings module for multimodal embedding models.

This module provides abstract base classes and concrete implementations
for multimodal embedding models that map text, images, or combinations
into a unified vector space.
"""

from openadapt_retrieval.embeddings.base import BaseEmbedder
from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder
from openadapt_retrieval.embeddings.clip import CLIPEmbedder
from openadapt_retrieval.embeddings.registry import get_embedder, EMBEDDER_REGISTRY

__all__ = [
    "BaseEmbedder",
    "Qwen3VLEmbedder",
    "CLIPEmbedder",
    "get_embedder",
    "EMBEDDER_REGISTRY",
]
