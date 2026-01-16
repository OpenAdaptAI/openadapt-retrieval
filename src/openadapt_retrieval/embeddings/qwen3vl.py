"""Qwen3-VL-Embedding implementation.

This module provides the primary multimodal embedder using Alibaba's
Qwen3-VL-Embedding model, which supports text, image, and combined
embeddings with Matryoshka Representation Learning (MRL) for flexible
embedding dimensions.

References:
    - Paper: https://arxiv.org/abs/2601.04720
    - Model: https://huggingface.co/Alibaba-NLP/gte-Qwen2-VL-2B-instruct
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from openadapt_retrieval.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VLEmbedderConfig:
    """Configuration for Qwen3VLEmbedder.

    Attributes:
        model_name: HuggingFace model name or local path.
        embedding_dim: Target embedding dimension. If None, uses model's full dimension.
            Supported values depend on model (e.g., 64-2048 for 2B model).
            Uses Matryoshka Representation Learning (MRL) for dimension reduction.
        device: Device to use: "cuda", "cpu", "mps", or specific like "cuda:0".
            If None, auto-detects best available device.
        torch_dtype: Model dtype. Use float16 for GPU, float32 for CPU.
        use_flash_attention: Whether to use Flash Attention 2.
        max_image_size: Maximum image dimension. Larger images are resized.
        normalize_embeddings: Whether to L2-normalize embeddings.
        cache_dir: Directory for caching downloaded models.
        default_instruction: Default instruction prefix for retrieval queries.
    """

    model_name: str = "Alibaba-NLP/gte-Qwen2-VL-2B-instruct"
    embedding_dim: Optional[int] = 512
    device: Optional[str] = None
    torch_dtype: str = "float16"
    use_flash_attention: bool = True
    max_image_size: int = 1280
    normalize_embeddings: bool = True
    cache_dir: Optional[Path] = None
    default_instruction: str = "Retrieve demonstrations for GUI automation tasks."

    # Model-specific max dimensions
    _max_dims: dict = field(default_factory=lambda: {
        "Alibaba-NLP/gte-Qwen2-VL-2B-instruct": 2048,
        "Qwen/Qwen3-VL-Embedding-2B": 2048,
        "Qwen/Qwen3-VL-Embedding-8B": 4096,
    })


class Qwen3VLEmbedder(BaseEmbedder):
    """Multimodal embedder using Qwen3-VL-Embedding.

    This embedder maps text, images, and text+image combinations into
    a unified semantic vector space using Alibaba's Qwen VL embedding models.

    Features:
        - Supports text-only, image-only, and multimodal inputs
        - Matryoshka Representation Learning (MRL) for flexible dimensions
        - Lazy model loading to minimize startup time
        - GPU/CPU/MPS support with automatic device detection
        - Flash Attention 2 for faster inference (when available)

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

    Hardware Requirements:
        - 2B model: ~6-8 GB VRAM (FP16)
        - Inference: ~50-200ms per embedding on RTX 4090
    """

    def __init__(
        self,
        config: Optional[Qwen3VLEmbedderConfig] = None,
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen3-VL-Embedding embedder.

        Args:
            config: Configuration object. If None, uses defaults.
            embedding_dim: Override for embedding dimension.
            device: Override for device selection.
            model_name: Override for model name.
            **kwargs: Additional config overrides.
        """
        if config is None:
            config = Qwen3VLEmbedderConfig()

        # Apply explicit overrides
        if embedding_dim is not None:
            config.embedding_dim = embedding_dim
        if device is not None:
            config.device = device
        if model_name is not None:
            config.model_name = model_name

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._torch = None

        # Validate embedding_dim
        max_dim = config._max_dims.get(config.model_name, 2048)
        if config.embedding_dim is not None:
            if config.embedding_dim > max_dim:
                raise ValueError(
                    f"embedding_dim {config.embedding_dim} exceeds model maximum {max_dim}"
                )
            if config.embedding_dim < 64:
                raise ValueError("embedding_dim must be at least 64")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self.config.embedding_dim is not None:
            return self.config.embedding_dim
        return self.config._max_dims.get(self.config.model_name, 2048)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.config.model_name

    def _get_device(self):
        """Determine the best available device."""
        if self._device is not None:
            return self._device

        import torch

        self._torch = torch

        if self.config.device is not None:
            self._device = torch.device(self.config.device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(f"Using device: {self._device}")
        return self._device

    def _load_model(self) -> None:
        """Lazy-load the model and processor."""
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.config.model_name}")

        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.40.0 and torch>=2.0.0 are required for Qwen3VLEmbedder. "
                "Install with: uv add transformers torch"
            )

        self._torch = torch
        device = self._get_device()

        # Determine dtype
        if self.config.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Use float32 on CPU
        if device.type == "cpu":
            torch_dtype = torch.float32

        # Determine attention implementation
        attn_impl = None
        if self.config.use_flash_attention and device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.debug("flash-attn not installed, using default attention")

        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        if self.config.cache_dir:
            model_kwargs["cache_dir"] = str(self.config.cache_dir)

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        if device.type != "cuda":
            self._model = self._model.to(device)

        self._model.eval()

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )

        logger.info(f"Model loaded successfully on {device}")

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Image.Image:
        """Load and preprocess image."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
            elif str(image).startswith(("http://", "https://")):
                import requests
                response = requests.get(str(image), timeout=30)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize if too large
        max_size = self.config.max_image_size
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _embed(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        instruction: Optional[str] = None,
    ) -> NDArray[np.float32]:
        """Internal embedding method that handles all input combinations."""
        if text is None and image is None:
            raise ValueError("At least one of text or image must be provided")

        self._load_model()

        # Use default instruction if not provided
        if instruction is None:
            instruction = self.config.default_instruction

        # Build messages for Qwen format
        content = []

        if instruction:
            content.append({"type": "text", "text": instruction})

        img = None
        if image is not None:
            img = self._prepare_image(image)
            content.append({"type": "image", "image": img})

        if text:
            content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        # Process through model
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self._processor(
            text=[prompt],
            images=[img] if img is not None else None,
            return_tensors="pt",
        )

        model_inputs = {
            k: v.to(self._model.device) for k, v in model_inputs.items()
        }

        with self._torch.no_grad():
            outputs = self._model(**model_inputs, output_hidden_states=True)

            # Get last hidden state at EOS position
            hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)

            # Find EOS token position (last non-padding token)
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1)
                eos_positions = seq_lens - 1
            else:
                eos_positions = self._torch.tensor(
                    [hidden_states.size(1) - 1],
                    device=hidden_states.device,
                )

            # Extract EOS embeddings
            batch_indices = self._torch.arange(
                hidden_states.size(0),
                device=hidden_states.device,
            )
            embedding = hidden_states[batch_indices, eos_positions]

            # Apply MRL dimension reduction if needed
            if self.config.embedding_dim is not None:
                embedding = embedding[:, :self.config.embedding_dim]

            # Normalize
            if self.config.normalize_embeddings:
                embedding = self._torch.nn.functional.normalize(embedding, p=2, dim=-1)

            result = embedding.cpu().numpy().astype(np.float32)

        return result[0]  # Return single embedding

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed a text string."""
        return self._embed(text=text, image=None)

    def embed_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> NDArray[np.float32]:
        """Embed an image."""
        return self._embed(text=None, image=image)

    def embed_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
    ) -> NDArray[np.float32]:
        """Embed text and image together."""
        return self._embed(text=text, image=image)

    def embed_batch(
        self,
        inputs: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Embed multiple inputs.

        Note: Currently processes inputs sequentially. Future versions
        may implement true batching for efficiency.
        """
        from tqdm import tqdm

        if not inputs:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        embeddings = []
        iterator = tqdm(inputs, desc="Embedding") if show_progress and len(inputs) > 1 else inputs

        for inp in iterator:
            text = inp.get("text")
            image = inp.get("image")
            instruction = inp.get("instruction")

            if text is None and image is None:
                raise ValueError("Each input must have at least 'text' or 'image' key")

            emb = self._embed(text=text, image=image, instruction=instruction)
            embeddings.append(emb)

        return np.stack(embeddings, axis=0).astype(np.float32)

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        logger.info("Model unloaded")
