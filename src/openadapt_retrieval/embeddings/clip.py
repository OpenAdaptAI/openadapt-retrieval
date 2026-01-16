"""CLIP embedding implementation (fallback/baseline).

This module provides a lighter-weight alternative to Qwen3-VL-Embedding
using OpenAI's CLIP model. Useful when GPU memory is limited or as a
baseline for comparison.

Note: Requires the 'clip' extra: uv add openadapt-retrieval[clip]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from openadapt_retrieval.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class CLIPEmbedder(BaseEmbedder):
    """Multimodal embedder using CLIP.

    CLIP (Contrastive Language-Image Pre-training) provides a lighter-weight
    alternative to Qwen3-VL-Embedding with lower memory requirements.

    Features:
        - ~2GB VRAM vs ~8GB for Qwen3-VL
        - Faster inference
        - Good for quick prototyping or memory-constrained environments

    Limitations:
        - Lower quality than Qwen3-VL for GUI-specific tasks
        - Fixed embedding dimension (no MRL support)
        - Less nuanced multimodal understanding

    Example:
        >>> embedder = CLIPEmbedder()
        >>>
        >>> # Text embedding
        >>> text_emb = embedder.embed_text("Turn off Night Shift")
        >>>
        >>> # Image embedding
        >>> img_emb = embedder.embed_image("/path/to/screenshot.png")
        >>>
        >>> # Multimodal (uses both text and image)
        >>> mm_emb = embedder.embed_multimodal(
        ...     text="Turn off Night Shift",
        ...     image="/path/to/screenshot.png",
        ... )
    """

    DEFAULT_MODEL = "ViT-L-14"
    DEFAULT_PRETRAINED = "openai"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the CLIP embedder.

        Args:
            model_name: CLIP model architecture (e.g., "ViT-L-14", "ViT-B-32").
            pretrained: Pretrained weights to use (e.g., "openai", "laion2b_s32b_b82k").
            device: Device for inference. Auto-detected if None.
            normalize_embeddings: Whether to L2-normalize embeddings.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self._model_name = model_name
        self._pretrained = pretrained
        self._device_str = device
        self._normalize_embeddings = normalize_embeddings

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._torch = None
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return f"clip/{self._model_name}/{self._pretrained}"

    def _get_device(self):
        """Determine the best available device."""
        if self._device is not None:
            return self._device

        import torch

        self._torch = torch

        if self._device_str is not None:
            self._device = torch.device(self._device_str)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(f"CLIP using device: {self._device}")
        return self._device

    def _load_model(self) -> None:
        """Lazy-load the CLIP model."""
        if self._model is not None:
            return

        logger.info(f"Loading CLIP model: {self._model_name}")

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for CLIPEmbedder. "
                "Install with: uv add openadapt-retrieval[clip]"
            )

        device = self._get_device()

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name,
            pretrained=self._pretrained,
            device=device,
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model.eval()

        # Get embedding dimension from model
        with self._torch.no_grad():
            dummy_text = self._tokenizer(["test"]).to(device)
            text_features = self._model.encode_text(dummy_text)
            self._embedding_dim = text_features.shape[-1]

        logger.info(f"CLIP loaded: dim={self._embedding_dim}, device={device}")

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Image.Image:
        """Load and prepare image for CLIP."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
            elif str(image).startswith(("http://", "https://")):
                import io
                import requests
                response = requests.get(str(image), timeout=30)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return img

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a text string using CLIP."""
        self._load_model()

        with self._torch.no_grad():
            tokens = self._tokenizer([text]).to(self._device)
            text_features = self._model.encode_text(tokens)

            if self._normalize_embeddings:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            embedding = text_features.cpu().numpy().astype(np.float32)

        return embedding[0]

    def embed_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> NDArray[np.float32]:
        """Embed an image using CLIP."""
        self._load_model()

        img = self._prepare_image(image)
        img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            image_features = self._model.encode_image(img_tensor)

            if self._normalize_embeddings:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embedding = image_features.cpu().numpy().astype(np.float32)

        return embedding[0]

    def embed_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
    ) -> NDArray[np.float32]:
        """Embed text and image together.

        CLIP embeds text and images separately. For multimodal, we average
        the normalized text and image embeddings.
        """
        text_emb = self.embed_text(text)
        image_emb = self.embed_image(image)

        # Average the normalized embeddings
        combined = (text_emb + image_emb) / 2.0

        # Re-normalize
        if self._normalize_embeddings:
            combined = combined / np.linalg.norm(combined)

        return combined.astype(np.float32)

    def embed_batch(
        self,
        inputs: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Embed multiple inputs."""
        from tqdm import tqdm

        if not inputs:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        self._load_model()

        embeddings = []
        iterator = tqdm(inputs, desc="CLIP Embedding") if show_progress and len(inputs) > 1 else inputs

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

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._preprocess is not None:
            del self._preprocess
            self._preprocess = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        logger.info("CLIP model unloaded")
