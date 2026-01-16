"""Command-line interface for openadapt-retrieval.

Usage:
    uv run python -m openadapt_retrieval.cli embed --text "Turn off Night Shift"
    uv run python -m openadapt_retrieval.cli embed --image screenshot.png
    uv run python -m openadapt_retrieval.cli embed --text "Task" --image screenshot.png

    uv run python -m openadapt_retrieval.cli index --demo-dir /path/to/demos --output demo_index/
    uv run python -m openadapt_retrieval.cli search --index demo_index/ --text "query" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed text and/or image."""
    from openadapt_retrieval.embeddings import get_embedder

    if not args.text and not args.image:
        print("Error: At least one of --text or --image is required", file=sys.stderr)
        return 1

    embedder = get_embedder(
        name=args.embedder,
        embedding_dim=args.dim,
        device=args.device,
    )

    print(f"Using embedder: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.embedding_dim}")

    if args.text and args.image:
        print(f"Embedding multimodal: text='{args.text[:50]}...' image={args.image}")
        embedding = embedder.embed_multimodal(text=args.text, image=args.image)
    elif args.text:
        print(f"Embedding text: '{args.text[:50]}...'")
        embedding = embedder.embed_text(text=args.text)
    else:
        print(f"Embedding image: {args.image}")
        embedding = embedder.embed_image(image=args.image)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")

    if args.output:
        if args.output.endswith(".npy"):
            np.save(args.output, embedding)
            print(f"Saved to: {args.output}")
        elif args.output.endswith(".json"):
            with open(args.output, "w") as f:
                json.dump({"embedding": embedding.tolist()}, f)
            print(f"Saved to: {args.output}")
        else:
            np.save(args.output + ".npy", embedding)
            print(f"Saved to: {args.output}.npy")
    else:
        print(f"Embedding (first 10 values): {embedding[:10]}")

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build index from demo directory."""
    from openadapt_retrieval import MultimodalDemoRetriever

    demo_dir = Path(args.demo_dir)
    if not demo_dir.exists():
        print(f"Error: Demo directory not found: {demo_dir}", file=sys.stderr)
        return 1

    retriever = MultimodalDemoRetriever(
        embedder_name=args.embedder,
        embedding_dim=args.dim,
        device=args.device,
        index_path=args.output,
    )

    # Discover demos in directory
    # Expected structure: demo_dir/{demo_id}/
    #   - task.txt or metadata.json (contains task description)
    #   - screenshot.png or screenshots/000.png (representative screenshot)

    demo_count = 0
    for demo_path in sorted(demo_dir.iterdir()):
        if not demo_path.is_dir():
            continue

        demo_id = demo_path.name
        task = None
        screenshot = None

        # Try to find task description
        task_file = demo_path / "task.txt"
        metadata_file = demo_path / "metadata.json"

        if task_file.exists():
            task = task_file.read_text().strip()
        elif metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            task = metadata.get("task") or metadata.get("instruction") or metadata.get("goal")

        if not task:
            print(f"Warning: No task found for {demo_id}, skipping")
            continue

        # Try to find screenshot
        for screenshot_name in [
            "screenshot.png",
            "screenshot.jpg",
            "screenshots/000.png",
            "screenshots/001.png",
        ]:
            screenshot_path = demo_path / screenshot_name
            if screenshot_path.exists():
                screenshot = str(screenshot_path)
                break

        # Load additional metadata if available
        app_name = None
        domain = None
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            app_name = metadata.get("app_name")
            domain = metadata.get("domain")

        retriever.add_demo(
            demo_id=demo_id,
            task=task,
            screenshot=screenshot,
            app_name=app_name,
            domain=domain,
        )
        demo_count += 1
        print(f"Added demo: {demo_id}")

    if demo_count == 0:
        print("Error: No demos found in directory", file=sys.stderr)
        return 1

    print(f"\nBuilding index for {demo_count} demos...")
    retriever.build_index()

    retriever.save()
    print(f"\nIndex saved to: {args.output}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search the index."""
    from openadapt_retrieval import MultimodalDemoRetriever

    if not args.text and not args.image:
        print("Error: At least one of --text or --image is required", file=sys.stderr)
        return 1

    index_path = Path(args.index)
    if not (index_path / "index.json").exists():
        print(f"Error: Index not found at {index_path}", file=sys.stderr)
        return 1

    retriever = MultimodalDemoRetriever(
        embedder_name=args.embedder,
        embedding_dim=args.dim,
        device=args.device,
        index_path=index_path,
    )

    retriever.load()
    print(f"Loaded index with {len(retriever)} demos")

    print(f"\nSearching for: text='{args.text}', image={args.image}")
    results = retriever.retrieve(
        task=args.text or "",
        screenshot=args.image,
        top_k=args.top_k,
    )

    if not results:
        print("No results found")
        return 0

    print(f"\nTop {len(results)} results:")
    print("-" * 60)

    for result in results:
        print(f"\n{result.rank}. {result.demo.demo_id}")
        print(f"   Task: {result.demo.task[:80]}{'...' if len(result.demo.task) > 80 else ''}")
        print(f"   Score: {result.score:.4f} (embedding: {result.embedding_score:.4f})")
        if result.demo.app_name:
            print(f"   App: {result.demo.app_name}")
        if result.demo.screenshot_path:
            print(f"   Screenshot: {result.demo.screenshot_path}")

    if args.output:
        output_data = [
            {
                "rank": r.rank,
                "demo_id": r.demo.demo_id,
                "task": r.demo.task,
                "score": r.score,
                "embedding_score": r.embedding_score,
            }
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show index information."""
    from openadapt_retrieval.storage import EmbeddingStorage

    storage = EmbeddingStorage(args.index)
    info = storage.get_info()

    if not info.get("exists"):
        print(f"Error: No index found at {args.index}", file=sys.stderr)
        return 1

    print(f"Index: {info['path']}")
    print(f"  Embeddings: {info['embedding_count']}")
    print(f"  Dimension: {info['embedding_dim']}")
    print(f"  Created: {info.get('created_at', 'Unknown')}")
    print(f"  Schema: {info.get('schema_version', 'Unknown')}")

    if info.get("config"):
        print(f"  Config: {json.dumps(info['config'], indent=4)}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenAdapt Retrieval - Multimodal demo retrieval CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # embed command
    embed_parser = subparsers.add_parser("embed", help="Embed text and/or image")
    embed_parser.add_argument("--text", "-t", help="Text to embed")
    embed_parser.add_argument("--image", "-i", help="Image path to embed")
    embed_parser.add_argument("--output", "-o", help="Output file path")
    embed_parser.add_argument(
        "--embedder", "-e",
        default="qwen3vl",
        choices=["qwen3vl", "clip"],
        help="Embedder to use (default: qwen3vl)",
    )
    embed_parser.add_argument(
        "--dim", "-d",
        type=int,
        default=512,
        help="Embedding dimension (default: 512)",
    )
    embed_parser.add_argument(
        "--device",
        help="Device to use (cuda, cpu, mps)",
    )

    # index command
    index_parser = subparsers.add_parser("index", help="Build index from demo directory")
    index_parser.add_argument(
        "--demo-dir", "-d",
        required=True,
        help="Directory containing demos",
    )
    index_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for index",
    )
    index_parser.add_argument(
        "--embedder", "-e",
        default="qwen3vl",
        choices=["qwen3vl", "clip"],
        help="Embedder to use",
    )
    index_parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Embedding dimension",
    )
    index_parser.add_argument(
        "--device",
        help="Device to use",
    )

    # search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument(
        "--index", "-I",
        required=True,
        help="Index directory path",
    )
    search_parser.add_argument("--text", "-t", help="Query text")
    search_parser.add_argument("--image", "-i", help="Query image")
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    search_parser.add_argument("--output", "-o", help="Output JSON file")
    search_parser.add_argument(
        "--embedder", "-e",
        default="qwen3vl",
        choices=["qwen3vl", "clip"],
        help="Embedder to use (must match index)",
    )
    search_parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Embedding dimension (must match index)",
    )
    search_parser.add_argument(
        "--device",
        help="Device to use",
    )

    # info command
    info_parser = subparsers.add_parser("info", help="Show index information")
    info_parser.add_argument(
        "--index", "-I",
        required=True,
        help="Index directory path",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    commands = {
        "embed": cmd_embed,
        "index": cmd_index,
        "search": cmd_search,
        "info": cmd_info,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
