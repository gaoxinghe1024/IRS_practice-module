"""Check the delivered algorithm 2 files against DELIVERY_MANIFEST.json (standard library only).

    .venv/bin/python algorithm2_benchmark/verify_delivery.py            # verify
    .venv/bin/python algorithm2_benchmark/verify_delivery.py --write    # record a new delivery

Every file of this folder is covered except caches, the regenerable `work/` folders and the manifest itself.
Re-running a stage rewrites some records (run_metadata.json, reports, tests.txt), so verify before re-running.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_PARTS = {"__pycache__", "work", ".pytest_cache"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in SKIP_PARTS for part in rel.parts):
            continue
        if rel.as_posix() == "DELIVERY_MANIFEST.json" or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        yield path


def main(write: bool = False) -> None:
    manifest_path = ROOT / "DELIVERY_MANIFEST.json"
    if write:
        entries = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in files()]
        manifest_path.write_text(json.dumps({"schema": "algorithm2-benchmark-delivery-v1", "files": entries},
                                            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        if (not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]):
            failures.append(entry["path"])
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))
    print(f"PASS: {len(manifest['files'])} delivered files match the SHA-256 manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="record the current files as the delivery")
    main(parser.parse_args().write)
