"""Retriever module for multimodal demo retrieval.

This module provides the main retrieval interface for indexing and
searching demonstration recordings based on text and image similarity.
"""

from openadapt_retrieval.retriever.demo_retriever import (
    MultimodalDemoRetriever,
    DemoMetadata,
    RetrievalResult,
)
from openadapt_retrieval.retriever.index import VectorIndex
from openadapt_retrieval.retriever.reranker import CrossEncoderReranker

__all__ = [
    "MultimodalDemoRetriever",
    "DemoMetadata",
    "RetrievalResult",
    "VectorIndex",
    "CrossEncoderReranker",
]
