"""Vector index for similarity search.

This module provides a wrapper around FAISS for efficient
vector similarity search with save/load support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-based vector index for similarity search.

    This class wraps FAISS to provide efficient nearest neighbor search
    with support for saving and loading indices.

    Features:
        - Automatic FAISS index type selection based on size
        - Inner product (cosine similarity for normalized vectors)
        - Save/load support for persistence
        - Falls back to numpy brute-force if FAISS unavailable

    Example:
        >>> index = VectorIndex(dimension=512)
        >>>
        >>> # Add vectors
        >>> index.add(embeddings)  # shape: (n_vectors, 512)
        >>>
        >>> # Search
        >>> scores, indices = index.search(query_vector, top_k=5)
        >>>
        >>> # Save and load
        >>> index.save("/path/to/index.faiss")
        >>> index.load("/path/to/index.faiss")
    """

    def __init__(
        self,
        dimension: int,
        use_faiss: bool = True,
        index_type: str = "flat",
    ) -> None:
        """Initialize the vector index.

        Args:
            dimension: Dimensionality of vectors.
            use_faiss: Whether to use FAISS (falls back to numpy if False or unavailable).
            index_type: FAISS index type. Options:
                - "flat": Exact search (IndexFlatIP)
                - "ivf": Approximate search (IndexIVFFlat) - for large indices
        """
        self.dimension = dimension
        self._use_faiss = use_faiss
        self._index_type = index_type
        self._faiss_index = None
        self._vectors: Optional[NDArray[np.float32]] = None
        self._faiss = None

        if use_faiss:
            self._init_faiss()

    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
            self._faiss = faiss
            logger.debug("FAISS available, using FAISS index")
        except ImportError:
            logger.warning("FAISS not available, falling back to numpy brute-force search")
            self._use_faiss = False

    def _create_faiss_index(self, n_vectors: int = 0) -> None:
        """Create the appropriate FAISS index based on settings and data size."""
        if not self._use_faiss or self._faiss is None:
            return

        if self._index_type == "ivf" and n_vectors >= 1000:
            # IVF index for larger datasets
            nlist = min(100, n_vectors // 10)  # Number of clusters
            quantizer = self._faiss.IndexFlatIP(self.dimension)
            self._faiss_index = self._faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, self._faiss.METRIC_INNER_PRODUCT
            )
        else:
            # Flat index for exact search (default)
            self._faiss_index = self._faiss.IndexFlatIP(self.dimension)

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self._faiss_index is not None:
            return self._faiss_index.ntotal
        elif self._vectors is not None:
            return len(self._vectors)
        return 0

    @property
    def is_trained(self) -> bool:
        """Return whether the index is trained (for IVF indices)."""
        if self._faiss_index is None:
            return True
        return getattr(self._faiss_index, "is_trained", True)

    def add(
        self,
        vectors: NDArray[np.float32],
    ) -> None:
        """Add vectors to the index.

        Args:
            vectors: Vectors to add, shape (n_vectors, dimension).

        Raises:
            ValueError: If vector dimension doesn't match index dimension.
        """
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
            )

        # Store vectors for numpy fallback
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        # Add to FAISS index
        if self._use_faiss and self._faiss is not None:
            if self._faiss_index is None:
                self._create_faiss_index(len(vectors))

            # Train IVF index if needed
            if not self.is_trained:
                self._faiss_index.train(self._vectors)

            self._faiss_index.add(vectors)

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int = 5,
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search for nearest neighbors.

        Args:
            query: Query vector(s), shape (dimension,) or (n_queries, dimension).
            top_k: Number of results to return.

        Returns:
            Tuple of (scores, indices):
                - scores: Similarity scores, shape (n_queries, top_k)
                - indices: Vector indices, shape (n_queries, top_k)

        Raises:
            ValueError: If index is empty.
        """
        if self.size == 0:
            raise ValueError("Index is empty. Add vectors before searching.")

        query = np.asarray(query, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)
            squeeze_result = True
        else:
            squeeze_result = False

        top_k = min(top_k, self.size)

        if self._faiss_index is not None:
            # FAISS search
            scores, indices = self._faiss_index.search(query, top_k)
        else:
            # Numpy brute-force search (inner product)
            similarities = query @ self._vectors.T  # (n_queries, n_vectors)
            indices = np.argsort(-similarities, axis=1)[:, :top_k]
            scores = np.take_along_axis(similarities, indices, axis=1)

        scores = scores.astype(np.float32)
        indices = indices.astype(np.int64)

        if squeeze_result:
            scores = scores[0]
            indices = indices[0]

        return scores, indices

    def save(self, path: Union[str, Path]) -> None:
        """Save the index to disk.

        Args:
            path: Path to save the index file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(path))
            logger.info(f"FAISS index saved to {path}")
        else:
            # Save numpy vectors
            np.save(str(path.with_suffix(".npy")), self._vectors)
            logger.info(f"Numpy vectors saved to {path.with_suffix('.npy')}")

    def load(self, path: Union[str, Path]) -> None:
        """Load the index from disk.

        Args:
            path: Path to load the index file from.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        path = Path(path)

        if self._faiss is not None and path.exists():
            self._faiss_index = self._faiss.read_index(str(path))
            self._vectors = None  # FAISS manages vectors
            logger.info(f"FAISS index loaded from {path}: {self._faiss_index.ntotal} vectors")
        elif path.with_suffix(".npy").exists():
            self._vectors = np.load(str(path.with_suffix(".npy")))
            self._faiss_index = None
            logger.info(f"Numpy vectors loaded from {path.with_suffix('.npy')}: {len(self._vectors)} vectors")
        else:
            raise FileNotFoundError(f"No index found at {path}")

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._vectors = None
        self._faiss_index = None
        if self._use_faiss:
            self._create_faiss_index()

    def get_vectors(self) -> Optional[NDArray[np.float32]]:
        """Get all vectors in the index.

        Returns:
            All vectors or None if using FAISS-only storage.
        """
        if self._vectors is not None:
            return self._vectors
        elif self._faiss_index is not None:
            # Reconstruct from FAISS if possible
            try:
                n = self._faiss_index.ntotal
                if n > 0:
                    return self._faiss_index.reconstruct_n(0, n)
            except Exception:
                pass
        return None
