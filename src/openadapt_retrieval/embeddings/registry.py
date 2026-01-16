"""Embedder registry and factory function.

This module provides a factory function to get embedders by name,
making it easy to switch between different embedding models.
"""

from __future__ import annotations

from typing import Any, Type

from openadapt_retrieval.embeddings.base import BaseEmbedder


# Registry of available embedders
EMBEDDER_REGISTRY: dict[str, Type[BaseEmbedder]] = {}


def _register_embedders() -> None:
    """Register built-in embedders."""
    from openadapt_retrieval.embeddings.qwen3vl import Qwen3VLEmbedder
    from openadapt_retrieval.embeddings.clip import CLIPEmbedder

    EMBEDDER_REGISTRY["qwen3vl"] = Qwen3VLEmbedder
    EMBEDDER_REGISTRY["qwen"] = Qwen3VLEmbedder  # Alias
    EMBEDDER_REGISTRY["clip"] = CLIPEmbedder


def get_embedder(
    name: str = "qwen3vl",
    **kwargs: Any,
) -> BaseEmbedder:
    """Get an embedder by name.

    This factory function provides a convenient way to instantiate
    embedders by name, without needing to import specific classes.

    Args:
        name: Name of the embedder. Options:
            - "qwen3vl" or "qwen": Qwen3-VL-Embedding (recommended)
            - "clip": CLIP fallback (lighter weight)
        **kwargs: Arguments passed to the embedder constructor.

    Returns:
        Initialized embedder instance.

    Raises:
        ValueError: If the embedder name is not recognized.

    Example:
        >>> # Get Qwen embedder with custom dimension
        >>> embedder = get_embedder("qwen3vl", embedding_dim=512)
        >>>
        >>> # Get CLIP embedder
        >>> embedder = get_embedder("clip")
        >>>
        >>> # Use the embedder
        >>> emb = embedder.embed_text("Turn off Night Shift")
    """
    # Ensure embedders are registered
    if not EMBEDDER_REGISTRY:
        _register_embedders()

    name_lower = name.lower()

    if name_lower not in EMBEDDER_REGISTRY:
        available = ", ".join(sorted(EMBEDDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown embedder: '{name}'. Available: {available}"
        )

    embedder_class = EMBEDDER_REGISTRY[name_lower]
    return embedder_class(**kwargs)


def register_embedder(name: str, embedder_class: Type[BaseEmbedder]) -> None:
    """Register a custom embedder.

    This allows extending the registry with custom embedder implementations.

    Args:
        name: Name to register the embedder under.
        embedder_class: The embedder class (must inherit from BaseEmbedder).

    Example:
        >>> from openadapt_retrieval.embeddings import register_embedder
        >>>
        >>> class MyCustomEmbedder(BaseEmbedder):
        ...     # Implementation
        ...     pass
        >>>
        >>> register_embedder("custom", MyCustomEmbedder)
        >>> embedder = get_embedder("custom")
    """
    if not issubclass(embedder_class, BaseEmbedder):
        raise TypeError(
            f"embedder_class must be a subclass of BaseEmbedder, got {embedder_class}"
        )

    EMBEDDER_REGISTRY[name.lower()] = embedder_class


def list_embedders() -> list[str]:
    """List available embedder names.

    Returns:
        List of registered embedder names.

    Example:
        >>> list_embedders()
        ['clip', 'qwen', 'qwen3vl']
    """
    if not EMBEDDER_REGISTRY:
        _register_embedders()

    return sorted(EMBEDDER_REGISTRY.keys())
