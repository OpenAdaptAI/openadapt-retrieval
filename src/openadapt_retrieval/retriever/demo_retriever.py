"""Multimodal demo retriever.

This module provides the main interface for indexing and retrieving
demonstration recordings based on text and image similarity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel

from openadapt_retrieval.embeddings import get_embedder, BaseEmbedder
from openadapt_retrieval.retriever.index import VectorIndex

logger = logging.getLogger(__name__)


class DemoMetadata(BaseModel):
    """Metadata for a demonstration recording.

    Attributes:
        demo_id: Unique identifier for the demo.
        task: Task description/instruction.
        screenshot_path: Path to the representative screenshot.
        app_name: Application name (e.g., 'System Settings').
        domain: Domain (e.g., 'github.com').
        platform: Operating system platform.
        tags: User-provided tags for categorization.
        metadata: Additional custom metadata.
    """

    demo_id: str
    task: str
    screenshot_path: Optional[str] = None
    app_name: Optional[str] = None
    domain: Optional[str] = None
    platform: Optional[str] = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}


class RetrievalResult(BaseModel):
    """A single retrieval result with score breakdown.

    Attributes:
        demo: The demo metadata.
        score: Combined retrieval score (higher is better).
        embedding_score: Raw embedding similarity.
        rerank_score: Reranker score (if reranking was used).
        rank: Rank in the result list (1-indexed).
    """

    demo: DemoMetadata
    score: float
    embedding_score: float
    rerank_score: Optional[float] = None
    rank: int = 0


@dataclass
class RetrieverConfig:
    """Configuration for MultimodalDemoRetriever.

    Attributes:
        embedder_name: Name of the embedder to use ('qwen3vl', 'clip').
        embedding_dim: Target embedding dimension (for MRL models).
        device: Device for inference ('cuda', 'cpu', 'mps', or None for auto).
        use_faiss: Whether to use FAISS for similarity search.
        index_path: Path for persisting the index.
        app_bonus: Bonus score for matching app name.
        domain_bonus: Bonus score for matching domain.
    """

    embedder_name: str = "qwen3vl"
    embedding_dim: Optional[int] = 512
    device: Optional[str] = None
    use_faiss: bool = True
    index_path: Optional[Path] = None
    app_bonus: float = 0.1
    domain_bonus: float = 0.1


class MultimodalDemoRetriever:
    """Demo retriever with multimodal (text + image) support.

    This retriever uses VLM embeddings to create joint text+image
    representations for demos and queries, enabling visual similarity
    matching in addition to semantic text matching.

    Features:
        - Multimodal embedding: combines task description + screenshot
        - FAISS-accelerated similarity search
        - Persistence to disk (embeddings + metadata)
        - Incremental index updates

    Example:
        >>> from openadapt_retrieval import MultimodalDemoRetriever
        >>>
        >>> # Initialize retriever
        >>> retriever = MultimodalDemoRetriever(embedding_dim=512)
        >>>
        >>> # Add demos
        >>> retriever.add_demo(
        ...     demo_id="turn-off-nightshift",
        ...     task="Turn off Night Shift in System Settings",
        ...     screenshot="/path/to/screenshot.png",
        ... )
        >>>
        >>> # Build the index
        >>> retriever.build_index()
        >>>
        >>> # Retrieve similar demos
        >>> results = retriever.retrieve(
        ...     task="Disable Night Shift",
        ...     screenshot="/path/to/current_screen.png",
        ...     top_k=3,
        ... )
        >>>
        >>> for result in results:
        ...     print(f"{result.demo.demo_id}: {result.score:.3f}")
    """

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        embedder_name: str = "qwen3vl",
        embedding_dim: Optional[int] = 512,
        device: Optional[str] = None,
        index_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the multimodal demo retriever.

        Args:
            config: Configuration object. If None, uses defaults with overrides.
            embedder_name: Override for embedder name.
            embedding_dim: Override for embedding dimension.
            device: Override for device selection.
            index_path: Override for index persistence path.
            **kwargs: Additional config overrides.
        """
        if config is None:
            config = RetrieverConfig()

        # Apply explicit overrides
        config.embedder_name = embedder_name
        if embedding_dim is not None:
            config.embedding_dim = embedding_dim
        if device is not None:
            config.device = device
        if index_path is not None:
            config.index_path = Path(index_path)

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Initialize embedder (lazy loaded)
        self._embedder: Optional[BaseEmbedder] = None

        # Index state
        self._demos: list[DemoMetadata] = []
        self._embeddings: Optional[NDArray[np.float32]] = None
        self._index: Optional[VectorIndex] = None
        self._is_indexed = False

    @property
    def embedder(self) -> BaseEmbedder:
        """Get or create the embedder (lazy initialization)."""
        if self._embedder is None:
            self._embedder = get_embedder(
                name=self.config.embedder_name,
                embedding_dim=self.config.embedding_dim,
                device=self.config.device,
            )
        return self._embedder

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedder.embedding_dim

    # =========================================================================
    # Demo Management
    # =========================================================================

    def add_demo(
        self,
        demo_id: str,
        task: str,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
        app_name: Optional[str] = None,
        domain: Optional[str] = None,
        platform: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DemoMetadata:
        """Add a demonstration to the library.

        Args:
            demo_id: Unique identifier for the demo.
            task: Task description/instruction.
            screenshot: Representative screenshot (path, URL, or PIL Image).
            app_name: Application name.
            domain: Domain (e.g., 'github.com').
            platform: Platform (e.g., 'macos', 'windows', 'web').
            tags: Tags for categorization.
            metadata: Additional custom metadata.

        Returns:
            DemoMetadata object for the added demo.
        """
        # Convert screenshot to path if it's a PIL Image
        screenshot_path = None
        if isinstance(screenshot, (str, Path)):
            screenshot_path = str(screenshot)
        elif screenshot is not None:
            # PIL Image - would need to save it; for now, store None
            logger.warning("PIL Image passed as screenshot; storing without path")

        demo = DemoMetadata(
            demo_id=demo_id,
            task=task,
            screenshot_path=screenshot_path,
            app_name=app_name,
            domain=domain,
            platform=platform,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._demos.append(demo)
        self._is_indexed = False  # Need to rebuild index

        return demo

    def add_demos(
        self,
        demos: list[dict[str, Any]],
    ) -> list[DemoMetadata]:
        """Add multiple demonstrations.

        Args:
            demos: List of demo dicts with keys matching add_demo() parameters.

        Returns:
            List of DemoMetadata objects.
        """
        return [self.add_demo(**demo) for demo in demos]

    def get_demo_count(self) -> int:
        """Get the number of demos in the library."""
        return len(self._demos)

    def get_all_demos(self) -> list[DemoMetadata]:
        """Get all demo metadata objects."""
        return list(self._demos)

    def get_demo(self, demo_id: str) -> Optional[DemoMetadata]:
        """Get a demo by ID."""
        for demo in self._demos:
            if demo.demo_id == demo_id:
                return demo
        return None

    def clear(self) -> None:
        """Clear all demos and reset the index."""
        self._demos = []
        self._embeddings = None
        self._index = None
        self._is_indexed = False

    # =========================================================================
    # Indexing
    # =========================================================================

    def build_index(self, force: bool = False) -> None:
        """Build the search index from all added demos.

        This computes multimodal embeddings for all demos and builds
        the vector index for similarity search.

        Args:
            force: If True, rebuild even if already indexed.

        Raises:
            ValueError: If no demos have been added.
        """
        if self._is_indexed and not force:
            logger.debug("Index already built, skipping (use force=True to rebuild)")
            return

        if not self._demos:
            raise ValueError("Cannot build index: no demos added. Use add_demo() first.")

        logger.info(f"Building multimodal index for {len(self._demos)} demos...")

        # Prepare batch inputs
        inputs = []
        for demo in self._demos:
            inp = {"text": demo.task}
            if demo.screenshot_path:
                inp["image"] = demo.screenshot_path
            inputs.append(inp)

        # Compute embeddings
        self._embeddings = self.embedder.embed_batch(inputs)

        # Build vector index
        self._index = VectorIndex(
            dimension=self.embedding_dim,
            use_faiss=self.config.use_faiss,
        )
        self._index.add(self._embeddings)

        self._is_indexed = True
        logger.info(f"Index built successfully with {len(self._demos)} demos")

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve(
        self,
        task: str,
        screenshot: Optional[Union[str, Path, Image.Image]] = None,
        top_k: int = 5,
        app_context: Optional[str] = None,
        domain_context: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-K most similar demos for a query.

        Args:
            task: Task description to find demos for.
            screenshot: Optional current screenshot for visual matching.
            top_k: Number of demos to retrieve.
            app_context: Optional app context for bonus scoring.
            domain_context: Optional domain context for bonus scoring.

        Returns:
            List of RetrievalResult objects, ordered by relevance.

        Raises:
            ValueError: If index has not been built.
        """
        if not self._is_indexed:
            raise ValueError("Index not built. Call build_index() first.")

        if not self._demos:
            return []

        # Create query embedding
        if screenshot is not None:
            query_embedding = self.embedder.embed_multimodal(text=task, image=screenshot)
        else:
            query_embedding = self.embedder.embed_text(text=task)

        # Search index
        top_k = min(top_k, len(self._demos))
        scores, indices = self._index.search(query_embedding, top_k=top_k)

        # Build results with context bonuses
        results = []
        for idx, score in zip(indices, scores):
            demo = self._demos[idx]
            bonus = self._compute_context_bonus(demo, app_context, domain_context)

            results.append(RetrievalResult(
                demo=demo,
                score=float(score) + bonus,
                embedding_score=float(score),
            ))

        # Sort by final score and assign ranks
        results.sort(key=lambda r: r.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _compute_context_bonus(
        self,
        demo: DemoMetadata,
        app_context: Optional[str],
        domain_context: Optional[str],
    ) -> float:
        """Compute context bonus for app/domain matching."""
        bonus = 0.0

        if app_context and demo.app_name:
            if app_context.lower() in demo.app_name.lower():
                bonus += self.config.app_bonus

        if domain_context and demo.domain:
            if domain_context.lower() in demo.domain.lower():
                bonus += self.config.domain_bonus

        return bonus

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the index to disk.

        Args:
            path: Directory to save index files. Uses config.index_path if None.

        Raises:
            ValueError: If no save path is specified.
        """
        if path is None:
            path = self.config.index_path
        if path is None:
            raise ValueError("No save path specified and config.index_path is None")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = []
        for demo in self._demos:
            metadata.append(demo.model_dump())

        # Save index config
        config_data = {
            "embedder_name": self.config.embedder_name,
            "embedding_dim": self.config.embedding_dim,
            "demos": metadata,
            "is_indexed": self._is_indexed,
        }

        with open(path / "index.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Save embeddings
        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)

        # Save FAISS index
        if self._index is not None:
            self._index.save(path / "faiss.index")

        logger.info(f"Index saved to {path} with {len(self._demos)} demos")

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load index from disk.

        Args:
            path: Directory containing index files. Uses config.index_path if None.

        Raises:
            ValueError: If no load path is specified.
            FileNotFoundError: If index files don't exist.
        """
        if path is None:
            path = self.config.index_path
        if path is None:
            raise ValueError("No load path specified and config.index_path is None")

        path = Path(path)

        with open(path / "index.json") as f:
            config_data = json.load(f)

        # Validate model compatibility
        if config_data.get("embedder_name") != self.config.embedder_name:
            logger.warning(
                f"Embedder mismatch: index uses {config_data.get('embedder_name')}, "
                f"current config uses {self.config.embedder_name}"
            )

        # Load demos
        self._demos = []
        for meta in config_data.get("demos", []):
            demo = DemoMetadata(**meta)
            self._demos.append(demo)

        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)

            # Recreate vector index
            self._index = VectorIndex(
                dimension=self._embeddings.shape[1],
                use_faiss=self.config.use_faiss,
            )
            self._index.add(self._embeddings)

            self._is_indexed = config_data.get("is_indexed", False)

        logger.info(f"Index loaded from {path} with {len(self._demos)} demos")

    # =========================================================================
    # Utilities
    # =========================================================================

    def __len__(self) -> int:
        """Return number of demos in the library."""
        return len(self._demos)

    def __repr__(self) -> str:
        """String representation."""
        status = "indexed" if self._is_indexed else "not indexed"
        return f"MultimodalDemoRetriever({len(self._demos)} demos, {self.config.embedder_name}, {status})"
