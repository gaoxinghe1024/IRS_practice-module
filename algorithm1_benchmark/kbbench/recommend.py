import argparse
import json
import sys

from .engine import METHODS, SimilarityEngine

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Independent Top-5 similarity demo; no backend/database needed")
    parser.add_argument("--anime-id", type=int, required=True)
    parser.add_argument("--method", choices=list(METHODS), default="A")
    parser.add_argument("--variant", choices=["baseline", "optimized"], default="optimized")
    args = parser.parse_args()
    engine = SimilarityEngine(variant=args.variant)
    if args.anime_id not in engine.index:
        parser.error("Unknown anime ID; use an ID from data/catalog.jsonl.gz")
    print(json.dumps(engine.recommend(args.anime_id, args.method), ensure_ascii=False, indent=2))
