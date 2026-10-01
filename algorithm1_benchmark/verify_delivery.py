"""Standard-library-only manifest verification; --write creates the release manifest."""
import argparse
from pathlib import Path

from kbbench.common import ROOT, read_json, sha256, write_json


def files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in {".cache", ".venv", "__pycache__"} for part in relative.parts):
            continue
        if relative.as_posix() == "DELIVERY_MANIFEST.json" or path.suffix == ".pyc":
            continue
        yield path


def main(write=False):
    manifest_path = ROOT / "DELIVERY_MANIFEST.json"
    if write:
        write_json(manifest_path, {"schema": "algorithm1-benchmark-delivery-v1", "files": [
            dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)) for path in files()]})
    manifest = read_json(manifest_path)
    failures = []
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            failures.append(entry["path"])
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))
    print(f"PASS: {len(manifest['files'])} delivered files match SHA-256 manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    main(parser.parse_args().write)
