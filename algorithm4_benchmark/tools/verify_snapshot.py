# -*- coding: utf-8 -*-
"""Compare algo/rag_snapshot.py with the system's rag.py.

Usage:
    python tools/verify_snapshot.py PATH_TO_SYSTEM_RAG_PY

The path is required: it is backend/services/rag.py of AnimeFinder v1.4.1 (git blob
3b5f6dc, for example at commit 426db20 of the team repository).

Every top-level function and constant of the snapshot is compared with the system
source after comments, docstrings and whitespace are removed, and reported as
identical or differing. The imports and the DB path differ on purpose (they carry the
`# SNAPSHOT-EDIT:` marker) and are skipped.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PKG_ROOT / "algo" / "rag_snapshot.py"


def _normalized(node) -> str:
    """ast.dump of a node with docstrings stripped - a comment/whitespace-free view."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
        node = ast.parse(ast.unparse(node))  # deep copy via round-trip
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) \
                    and n.body and isinstance(n.body[0], ast.Expr) \
                    and isinstance(n.body[0].value, ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
        return ast.dump(node)
    return ast.dump(node)


def top_level_defs(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = _normalized(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node.value)
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python tools/verify_snapshot.py <path/to/system/rag.py>")
        return 2
    system = Path(sys.argv[1])
    if not system.exists():
        print(f"system rag.py not found: {system}")
        print("pass the path explicitly: python tools/verify_snapshot.py <path/to/rag.py>")
        return 2

    snap = top_level_defs(SNAPSHOT)
    sysm = top_level_defs(system)

    # names allowed to differ: snapshot plumbing added by SNAPSHOT-EDIT
    ALLOW_DIFFER = {"DB_PATH", "get_conn"}
    SKIP_SYSTEM_ONLY = {"init_db", "load_mal_dataset", "_seed_ratings",
                        "ANIME_SEED", "SCENE_SEED", "_PREFERENCE_CLUSTERS",
                        "USER_SEED", "DB_PATH", "get_conn"}

    identical, differs, missing = [], [], []
    for name, src in snap.items():
        if name in ALLOW_DIFFER:
            continue
        if name not in sysm:
            missing.append(name)
        elif sysm[name] == src:
            identical.append(name)
        else:
            differs.append(name)

    print(f"snapshot: {SNAPSHOT}")
    print(f"system  : {system}")
    print(f"identical ({len(identical)}): {', '.join(identical)}")
    if differs:
        print(f"DIFFERS  ({len(differs)}): {', '.join(differs)}")
    if missing:
        print(f"MISSING in system ({len(missing)}): {', '.join(missing)}")
    ok = not differs and not missing
    print("VERIFY_SNAPSHOT_OK" if ok else "VERIFY_SNAPSHOT_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
