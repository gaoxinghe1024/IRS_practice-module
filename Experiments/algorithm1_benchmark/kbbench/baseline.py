import argparse
import json
import sys

from .common import ROOT, read_rows
from .features import DIMENSIONS, WEIGHTS, pair_features


def recommend(anime_id, k=5):
    if k < 1:
        raise ValueError("k must be positive")
    items = list(read_rows(ROOT / "data/catalog.jsonl.gz"))
    target = next((item for item in items if item["anime_id"] == anime_id), None)
    if target is None:
        raise ValueError("Unknown anime ID")
    result = []
    for item in items:
        if item["anime_id"] == anime_id:
            continue
        features = pair_features(target, item)
        result.append(dict(anime_id=item["anime_id"], title=item["title"],
                           similarity=sum(weight * feature for weight, feature in zip(WEIGHTS, features)),
                           dimensions=dict(zip(DIMENSIONS, features))))
    return sorted(result, key=lambda item: (-item["similarity"], item["anime_id"]))[:k]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Weighted five-dimension similarity")
    parser.add_argument("--anime-id", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(recommend(args.anime_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
