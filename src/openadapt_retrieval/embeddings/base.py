"""Base class for multimodal embedding models.

This module defines the abstract interface for all embedding models,
providing a consistent API for text, image, and multimodal embeddings.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class BaseEmbedder(ABC):
    """Abstract base class for multimodal embedding models.

    Multimodal embedders map text, images, or combinations into a unified
    vector space. All inputs produce vectors of the same dimension.

    Example:
        >>> embedder = Qwen3VLEmbedder(embedding_dim=512)
        >>>
        >>> # Text-only embedding
        >>> text_emb = embedder.embed_text("Turn off Night Shift")
        >>>
        >>> # Image-only embedding
        >>> img_emb = embedder.embed_image("/path/to/screenshot.png")
        >>>
        >>> # Multimodal embedding (recommended)
        >>> mm_emb = embedder.embed_multimodal(
        ...     text="Turn off Night Shift",
        ...     image="/path/to/screenshot.png",
        ... )
        >>>
        >>> # Compute similarity
        >>> similarity = embedder.cosine_similarity(text_emb, img_emb)
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The dimensionality of the embedding vectors produced by this model.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier.

        Returns:
            str: A unique identifier for this embedding model (e.g., HuggingFace model name).
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as float32 numpy array of shape (embedding_dim,).
        """
        pass

    @abstractmethod
    def embed_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> NDArray[np.float32]:
        """Embed an image.

        Args:
            image: Image to embed. Can be a file path, URL, or PIL Image.

        Returns:
            Embedding vector as float32 numpy array of shape (embedding_dim,).

        Raises:
            FileNotFoundError: If image path does not exist.
            ValueError: If image format is not supported.
        """
        pass

    @abstractmethod
    def embed_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
    ) -> NDArray[np.float32]:
        """Embed text and image together (recommended for best retrieval quality).

        Args:
            text: Text description or instruction.
            image: Image to embed alongside text.

        Returns:
            Embedding vector as float32 numpy array of shape (embedding_dim,).
        """
        pass

    def embed_batch(
        self,
        inputs: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Embed multiple inputs.

        Args:
            inputs: List of dicts with 'text' and/or 'image' keys.
                    Each dict should have at least one of these keys.
            show_progress: Whether to show progress bar.

        Returns:
            Embeddings matrix of shape (n_inputs, embedding_dim).

        Raises:
            ValueError: If any input lacks both 'text' and 'image' keys.
        """
        from tqdm import tqdm

        embeddings = []
        iterator = tqdm(inputs, desc="Embedding") if show_progress and len(inputs) > 1 else inputs

        for inp in iterator:
            text = inp.get("text")
            image = inp.get("image")

            if text is not None and image is not None:
                emb = self.embed_multimodal(text=text, image=image)
            elif text is not None:
                emb = self.embed_text(text=text)
            elif image is not None:
                emb = self.embed_image(image=image)
            else:
                raise ValueError("Each input must have at least 'text' or 'image' key")

            embeddings.append(emb)

        return np.stack(embeddings, axis=0).astype(np.float32)

    def cosine_similarity(
        self,
        vec1: NDArray[np.float32],
        vec2: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            Cosine similarity score in range [-1, 1].
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def normalize(self, embedding: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize an embedding vector.

        Args:
            embedding: Embedding vector to normalize.

        Returns:
            L2-normalized embedding vector.
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return (embedding / norm).astype(np.float32)
