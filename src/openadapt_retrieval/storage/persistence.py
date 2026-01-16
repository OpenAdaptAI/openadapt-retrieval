"""Embedding storage and persistence utilities.

This module provides utilities for saving and loading embeddings
along with their associated metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingStorage:
    """Storage manager for embeddings and metadata.

    This class handles serialization and deserialization of embeddings
    along with their metadata to/from disk.

    Storage Format:
        storage_dir/
        ├── index.json       # Metadata and configuration
        ├── embeddings.npy   # Embedding vectors (float32)
        └── metadata.json    # Per-embedding metadata (optional)

    Example:
        >>> storage = EmbeddingStorage("/path/to/storage")
        >>>
        >>> # Save embeddings
        >>> storage.save(
        ...     embeddings=embeddings,
        ...     metadata=[{"id": "demo1"}, {"id": "demo2"}],
        ...     config={"model": "qwen3vl", "dim": 512},
        ... )
        >>>
        >>> # Load embeddings
        >>> data = storage.load()
        >>> embeddings = data["embeddings"]
        >>> metadata = data["metadata"]
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        path: Union[str, Path],
    ) -> None:
        """Initialize the storage manager.

        Args:
            path: Directory for storing embeddings.
        """
        self.path = Path(path)

    def save(
        self,
        embeddings: NDArray[np.float32],
        metadata: Optional[list[dict[str, Any]]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save embeddings and metadata to disk.

        Args:
            embeddings: Embedding matrix, shape (n_embeddings, embedding_dim).
            metadata: Optional list of metadata dicts, one per embedding.
            config: Optional configuration dict (model name, dimension, etc.).

        Raises:
            ValueError: If metadata length doesn't match embeddings count.
        """
        self.path.mkdir(parents=True, exist_ok=True)

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if metadata is not None and len(metadata) != len(embeddings):
            raise ValueError(
                f"Metadata length ({len(metadata)}) doesn't match "
                f"embeddings count ({len(embeddings)})"
            )

        # Save embeddings
        np.save(self.path / "embeddings.npy", embeddings)

        # Save index configuration
        index_data = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_count": len(embeddings),
            "embedding_dim": embeddings.shape[1] if embeddings.ndim == 2 else 0,
            "config": config or {},
        }

        with open(self.path / "index.json", "w") as f:
            json.dump(index_data, f, indent=2)

        # Save metadata if provided
        if metadata is not None:
            with open(self.path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved {len(embeddings)} embeddings to {self.path} "
            f"(dim={embeddings.shape[1] if embeddings.ndim == 2 else 'N/A'})"
        )

    def load(self) -> dict[str, Any]:
        """Load embeddings and metadata from disk.

        Returns:
            Dict containing:
                - embeddings: NDArray[np.float32]
                - metadata: Optional[list[dict]]
                - config: dict
                - index: dict (full index.json contents)

        Raises:
            FileNotFoundError: If storage doesn't exist.
        """
        if not (self.path / "index.json").exists():
            raise FileNotFoundError(f"No index found at {self.path}")

        # Load index
        with open(self.path / "index.json") as f:
            index_data = json.load(f)

        # Load embeddings
        embeddings_path = self.path / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
        else:
            embeddings = np.array([], dtype=np.float32)

        # Load metadata if exists
        metadata_path = self.path / "metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        logger.info(
            f"Loaded {len(embeddings)} embeddings from {self.path}"
        )

        return {
            "embeddings": embeddings,
            "metadata": metadata,
            "config": index_data.get("config", {}),
            "index": index_data,
        }

    def append(
        self,
        embeddings: NDArray[np.float32],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Append new embeddings to existing storage.

        Args:
            embeddings: New embeddings to append.
            metadata: Optional metadata for new embeddings.

        Note:
            This is a convenience method that loads existing data,
            appends new data, and saves everything. For large datasets,
            consider using incremental storage formats.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        try:
            existing = self.load()
            existing_emb = existing["embeddings"]
            existing_meta = existing["metadata"]
            config = existing["config"]

            # Append embeddings
            if len(existing_emb) > 0:
                new_embeddings = np.vstack([existing_emb, embeddings])
            else:
                new_embeddings = embeddings

            # Append metadata
            new_metadata = None
            if existing_meta is not None or metadata is not None:
                new_metadata = (existing_meta or []) + (metadata or [])

            self.save(new_embeddings, new_metadata, config)

        except FileNotFoundError:
            # No existing storage, just save
            self.save(embeddings, metadata)

    def exists(self) -> bool:
        """Check if storage exists."""
        return (self.path / "index.json").exists()

    def get_info(self) -> dict[str, Any]:
        """Get storage information without loading embeddings.

        Returns:
            Dict with count, dimension, creation time, etc.
        """
        if not self.exists():
            return {"exists": False}

        with open(self.path / "index.json") as f:
            index_data = json.load(f)

        return {
            "exists": True,
            "path": str(self.path),
            "embedding_count": index_data.get("embedding_count", 0),
            "embedding_dim": index_data.get("embedding_dim", 0),
            "created_at": index_data.get("created_at"),
            "schema_version": index_data.get("schema_version"),
            "config": index_data.get("config", {}),
        }

    def delete(self) -> None:
        """Delete the storage directory and all contents."""
        import shutil

        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info(f"Deleted storage at {self.path}")
