# Claude Code Instructions for openadapt-retrieval

## Overview

`openadapt-retrieval` provides multimodal demo retrieval for GUI automation using VLM embeddings.

## Quick Commands

```bash
# Install dependencies
cd /Users/abrichr/oa/src/openadapt-retrieval
uv sync

# Install with optional dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_embeddings.py -v

# Run CLI commands
uv run python -m openadapt_retrieval.cli --help
uv run python -m openadapt_retrieval.cli embed --text "Turn off Night Shift"
uv run python -m openadapt_retrieval.cli embed --image screenshot.png
uv run python -m openadapt_retrieval.cli index --demo-dir /path/to/demos --output demo_index/
uv run python -m openadapt_retrieval.cli search --index demo_index/ --text "query" --top-k 5
```

## Package Structure

```
openadapt-retrieval/
├── src/openadapt_retrieval/
│   ├── __init__.py           # Package exports
│   ├── cli.py                # CLI commands
│   ├── embeddings/
│   │   ├── base.py           # BaseEmbedder ABC
│   │   ├── qwen3vl.py        # Qwen3VLEmbedder (primary)
│   │   ├── clip.py           # CLIPEmbedder (fallback)
│   │   └── registry.py       # get_embedder() factory
│   ├── retriever/
│   │   ├── demo_retriever.py # MultimodalDemoRetriever
│   │   ├── index.py          # VectorIndex (FAISS wrapper)
│   │   └── reranker.py       # CrossEncoderReranker
│   └── storage/
│       └── persistence.py    # EmbeddingStorage
└── tests/
    ├── test_embeddings.py
    └── test_retriever.py
```

## Key Classes

### BaseEmbedder
Abstract base class for all embedders:
- `embed_text(text) -> np.ndarray` - Embed text only
- `embed_image(image) -> np.ndarray` - Embed image only
- `embed_multimodal(text, image) -> np.ndarray` - Embed both (recommended)
- `embed_batch(inputs) -> np.ndarray` - Batch embedding
- `cosine_similarity(vec1, vec2) -> float` - Similarity metric

### Qwen3VLEmbedder
Primary embedder using Qwen3-VL-Embedding:
- Supports MRL (Matryoshka Representation Learning) for flexible dimensions
- Lazy model loading to minimize startup time
- GPU/CPU/MPS support with automatic device detection

### CLIPEmbedder
Lighter-weight fallback using CLIP:
- ~2GB VRAM vs ~8GB for Qwen3-VL
- Requires `uv sync --extra clip`

### MultimodalDemoRetriever
Main retrieval interface:
- `add_demo(demo_id, task, screenshot, ...)` - Add demo to library
- `build_index()` - Compute embeddings and build FAISS index
- `retrieve(task, screenshot, top_k)` - Find similar demos
- `save(path)` / `load(path)` - Persistence

### VectorIndex
FAISS wrapper for similarity search:
- Automatic FAISS/numpy fallback
- Save/load support
- Inner product (cosine similarity for normalized vectors)

## Usage Patterns

### Basic Retrieval

```python
from openadapt_retrieval import MultimodalDemoRetriever

# Initialize
retriever = MultimodalDemoRetriever(embedding_dim=512)

# Add demos
retriever.add_demo(
    demo_id="turn-off-nightshift",
    task="Turn off Night Shift in System Settings",
    screenshot="/path/to/screenshot.png",
    app_name="System Settings",
)

# Build index
retriever.build_index()

# Retrieve
results = retriever.retrieve(
    task="Disable Night Shift",
    screenshot="/path/to/current_screen.png",
    top_k=3,
)
```

### Direct Embedding

```python
from openadapt_retrieval.embeddings import get_embedder

embedder = get_embedder("qwen3vl", embedding_dim=512)

# Text embedding
text_emb = embedder.embed_text("Turn off Night Shift")

# Image embedding
img_emb = embedder.embed_image("/path/to/screenshot.png")

# Multimodal embedding
mm_emb = embedder.embed_multimodal(
    text="Turn off Night Shift",
    image="/path/to/screenshot.png",
)
```

## Hardware Requirements

### Qwen3-VL-Embedding
- GPU: RTX 3060 (12GB) minimum, RTX 4090 (24GB) recommended
- VRAM: ~6-8 GB (FP16)
- Inference: ~50-200ms per embedding

### CLIP Fallback
- GPU: RTX 2060 (6GB) or better
- VRAM: ~2 GB
- Inference: ~10-50ms per embedding

### CPU-Only
Works but slower (~10x):
```python
embedder = Qwen3VLEmbedder(device="cpu", embedding_dim=256)
```

## Configuration

### Embedding Dimensions (MRL)

```python
# Full dimension (best quality, larger storage)
embedder = Qwen3VLEmbedder(embedding_dim=None)  # Uses 2048

# Reduced dimensions (faster search, smaller storage)
embedder = Qwen3VLEmbedder(embedding_dim=512)   # Good balance
embedder = Qwen3VLEmbedder(embedding_dim=256)   # Faster
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

## Testing

```bash
# Run all tests (uses mocks, no GPU needed)
uv run pytest tests/ -v

# Run GPU tests (requires CUDA)
uv run pytest tests/ -v -m gpu

# Run with coverage
uv run pytest tests/ --cov=openadapt_retrieval
```

## Dependencies

Core dependencies (always installed):
- torch, transformers - Model loading and inference
- pillow - Image processing
- faiss-cpu - Vector similarity search
- numpy, tqdm - Utilities
- pydantic - Data validation

Optional dependencies:
- `[gpu]`: faiss-gpu, accelerate
- `[clip]`: open-clip-torch
- `[serve]`: fastapi, uvicorn
- `[dev]`: pytest, ruff

## Related Projects

- [openadapt-ml](https://github.com/OpenAdaptAI/openadapt-ml) - ML engine for GUI automation
- [openadapt-grounding](https://github.com/OpenAdaptAI/openadapt-grounding) - UI element localization
- [openadapt-capture](https://github.com/OpenAdaptAI/openadapt-capture) - Recording capture

## References

- [Qwen3-VL-Embedding Paper](https://arxiv.org/abs/2601.04720)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
