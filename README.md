# openadapt-retrieval

Multimodal demo retrieval using VLM embeddings for GUI automation.

## Overview

`openadapt-retrieval` provides a unified interface for creating multimodal embeddings from screenshots and task descriptions, enabling semantic demo retrieval for GUI automation agents.

**Key Features:**
- **Multimodal Embeddings**: Embed text, images, or both into a shared vector space
- **Qwen3-VL-Embedding Support**: Primary embedder using Alibaba's state-of-the-art VLM
- **Matryoshka Representation Learning (MRL)**: Flexible embedding dimensions (512-8192)
- **FAISS Integration**: Fast similarity search with support for large demo libraries
- **Persistence**: Save and load indices with embeddings and metadata
- **CLI Interface**: Easy command-line access for indexing and searching

## Installation

```bash
# Basic installation
pip install openadapt-retrieval

# With GPU support
pip install openadapt-retrieval[gpu]

# With CLIP fallback embedder
pip install openadapt-retrieval[clip]

# All optional dependencies
pip install openadapt-retrieval[all]
```

For development:
```bash
git clone https://github.com/OpenAdaptAI/openadapt-retrieval.git
cd openadapt-retrieval
uv sync --all-extras
```

## Quick Start

### Python API

```python
from openadapt_retrieval import MultimodalDemoRetriever, Qwen3VLEmbedder

# Initialize retriever
retriever = MultimodalDemoRetriever(
    embedding_dim=512,  # Use MRL for smaller storage
)

# Add demos (from your recording library)
for demo in demos:
    retriever.add_demo(
        demo_id=demo.id,
        task=demo.instruction,
        screenshot=demo.first_screenshot_path,
        metadata={"app": demo.app_name},
    )

# Build the index
retriever.build_index()

# Save for later use
retriever.save("/path/to/demo_index")

# Retrieve similar demos
results = retriever.retrieve(
    task="Disable Night Shift",
    screenshot="/path/to/current_screen.png",
    top_k=3,
)

for result in results:
    print(f"{result.demo_id}: {result.task} (score: {result.score:.3f})")
```

### Using the Embedder Directly

```python
from openadapt_retrieval.embeddings import Qwen3VLEmbedder

# Initialize embedder
embedder = Qwen3VLEmbedder(embedding_dim=512)

# Embed text only
text_emb = embedder.embed_text("Turn off Night Shift")

# Embed image only
img_emb = embedder.embed_image("/path/to/screenshot.png")

# Embed multimodal (recommended)
mm_emb = embedder.embed_multimodal(
    text="Turn off Night Shift",
    image="/path/to/screenshot.png",
)

# Compute similarity
similarity = embedder.cosine_similarity(query_emb, demo_emb)
```

### CLI Usage

```bash
# Embed a single image
openadapt-retrieval embed --image screenshot.png --output embedding.npy

# Embed text + image
openadapt-retrieval embed --text "Turn off Night Shift" --image screenshot.png

# Build index from directory of demos
openadapt-retrieval index --demo-dir /path/to/demos --output demo_index/

# Search the index
openadapt-retrieval search --index demo_index/ --text "disable display setting" --top-k 5

# Search with screenshot
openadapt-retrieval search --index demo_index/ --text "disable display" --image current.png --top-k 3
```

## Architecture

### Embeddings Module

```
openadapt_retrieval/embeddings/
├── base.py       # BaseEmbedder abstract class
├── qwen3vl.py    # Qwen3-VL-Embedding implementation
├── clip.py       # CLIP fallback (lighter weight)
└── registry.py   # get_embedder() factory
```

**Supported Models:**
| Model | Embedding Dim | VRAM | Use Case |
|-------|--------------|------|----------|
| `Alibaba-NLP/Qwen3-VL-Embedding` | 512-8192 (MRL) | ~8GB | Primary (best quality) |
| `openai/clip-vit-large-patch14` | 768 | ~2GB | Fallback (lighter) |

### Retriever Module

```
openadapt_retrieval/retriever/
├── demo_retriever.py  # MultimodalDemoRetriever
├── index.py           # VectorIndex (FAISS wrapper)
└── reranker.py        # CrossEncoderReranker (optional)
```

**Key Classes:**
- `MultimodalDemoRetriever`: Main interface for indexing and retrieving demos
- `VectorIndex`: FAISS index wrapper with save/load support
- `CrossEncoderReranker`: Optional two-stage retrieval with cross-attention

### Storage Module

```
openadapt_retrieval/storage/
└── persistence.py  # EmbeddingStorage for save/load
```

**Index Format:**
```
demo_index/
├── index.json       # Metadata and configuration
├── embeddings.npy   # Embedding vectors (float32)
└── faiss.index      # FAISS index (optional, for large indices)
```

## Configuration

### Embedding Dimensions (MRL)

Qwen3-VL-Embedding supports Matryoshka Representation Learning for flexible dimensions:

```python
# Full dimension (best quality)
embedder = Qwen3VLEmbedder(embedding_dim=None)  # Uses 8192 for full model

# Reduced dimensions (faster search, smaller storage)
embedder = Qwen3VLEmbedder(embedding_dim=512)   # Good balance
embedder = Qwen3VLEmbedder(embedding_dim=256)   # Faster, slightly lower quality
```

### Device Selection

```python
# Auto-detect (CUDA > MPS > CPU)
embedder = Qwen3VLEmbedder()

# Force specific device
embedder = Qwen3VLEmbedder(device="cuda:0")
embedder = Qwen3VLEmbedder(device="mps")  # Apple Silicon
embedder = Qwen3VLEmbedder(device="cpu")
```

## Hardware Requirements

### Qwen3-VL-Embedding

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) |
| VRAM | 6 GB (FP16) | 8 GB |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB (model cache) | 20 GB |

### CPU-Only Mode

For machines without GPU, the embedder falls back to CPU (slower but functional):

```python
embedder = Qwen3VLEmbedder(device="cpu", embedding_dim=256)  # Smaller dim for speed
```

### Apple Silicon (MPS)

Native support for M1/M2/M3 Macs:

```python
embedder = Qwen3VLEmbedder(device="mps")
```

Performance: ~200-500ms per embedding depending on chip.

## Performance

| Operation | Demo Count | Time (RTX 4090) | Time (CPU) |
|-----------|------------|-----------------|------------|
| Embed 1 demo | 1 | ~200ms | ~2s |
| Embed 100 demos | 100 | ~15s | ~3min |
| Query (text+image) | any | ~150ms | ~2s |

## API Reference

### MultimodalDemoRetriever

```python
class MultimodalDemoRetriever:
    def __init__(
        self,
        embedding_model: str = "Alibaba-NLP/Qwen3-VL-Embedding",
        embedding_dim: int = 512,
        device: str | None = None,
        index_path: str | Path | None = None,
    ): ...

    def add_demo(
        self,
        demo_id: str,
        task: str,
        screenshot: str | Path | Image.Image | None = None,
        metadata: dict | None = None,
    ) -> None: ...

    def build_index(self, force: bool = False) -> None: ...

    def retrieve(
        self,
        task: str,
        screenshot: str | Path | Image.Image | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]: ...

    def save(self, path: str | Path | None = None) -> None: ...
    def load(self, path: str | Path | None = None) -> None: ...
```

### BaseEmbedder

```python
class BaseEmbedder(ABC):
    @property
    def embedding_dim(self) -> int: ...
    @property
    def model_name(self) -> str: ...

    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_image(self, image: str | Path | Image.Image) -> np.ndarray: ...
    def embed_multimodal(self, text: str, image: str | Path | Image.Image) -> np.ndarray: ...
    def embed_batch(self, inputs: list[dict]) -> np.ndarray: ...
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float: ...
```

## Related Projects

- [openadapt-ml](https://github.com/OpenAdaptAI/openadapt-ml) - ML engine for GUI automation
- [openadapt-grounding](https://github.com/OpenAdaptAI/openadapt-grounding) - UI element localization
- [openadapt-evals](https://github.com/OpenAdaptAI/openadapt-evals) - Benchmark evaluation infrastructure
- [openadapt-viewer](https://github.com/OpenAdaptAI/openadapt-viewer) - Dashboard visualization

## References

- [Qwen3-VL-Embedding Paper](https://arxiv.org/abs/2601.04720)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)

## License

MIT License - see LICENSE file for details.
