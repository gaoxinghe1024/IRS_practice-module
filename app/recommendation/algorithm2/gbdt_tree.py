"""Pure-NumPy evaluation of a saved LightGBM model (text format), used by the web service.

Why: on macOS, LightGBM's native library needs Homebrew's OpenMP runtime while PyTorch ships its own;
loading both into one process hangs or crashes. The service (which loads PyTorch for retrieval and the
two-tower model) therefore reads the LightGBM model file and evaluates the trees in NumPy. Training and
the offline evaluation use native LightGBM in processes without PyTorch; `freeze` records that both
give the same raw scores (tests check it too).

Decision rule per node (LightGBM `Tree::NumericalDecision`):
    missing_type = (decision_type >> 2) & 3            0 none, 1 zero, 2 NaN
    if value is NaN and missing_type != NaN: value = 0.0
    if (missing_type == zero and |value| <= kZeroThreshold) or (missing_type == NaN and value is NaN):
        go to the default child (left when decision_type & 2)
    else: go left when value <= threshold
A negative child index c denotes leaf ~c. The raw score is the sum of the reached leaf values.
Categorical splits are not supported (algorithm 2 has numeric features only) and raise an error.

Evaluation (QuickScorer-style bit masks, `predict_raw`): the leaves of every tree get bit positions in
left-to-right order and every split node a mask that clears the leaves of its left subtree. A row goes
right at a node exactly when its value exceeds the node's threshold, so per feature the right-going nodes
are a prefix of that feature's thresholds sorted ascending; prefix-AND tables (one row per prefix, one
column per tree, plus a row for NaN) give each row's per-tree mask with one gather per feature. The exit
leaf is the lowest bit left set. Tree outputs are then added in model order, as LightGBM does, so the raw
scores are identical. Models this does not cover (zero-as-missing splits, more than 64 leaves) use the
node-by-node `predict_levelwise`, which is also the reference implementation in the tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

K_ZERO_THRESHOLD = float(np.float32(1e-35))


@dataclass
class _Tree:
    split_feature: np.ndarray
    threshold: np.ndarray
    default_left: np.ndarray
    missing_type: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf_value: np.ndarray
    depth: int


def _depth(left: np.ndarray, right: np.ndarray) -> int:
    best, stack = 0, [(0, 1)]
    while stack:
        node, d = stack.pop()
        best = max(best, d)
        for c in (left[node], right[node]):
            if c >= 0:
                stack.append((int(c), d + 1))
    return best


def _inorder_leaves(t: _Tree, c: int) -> list[int]:
    """Leaf ids under child pointer c (node index >= 0, leaf ~id < 0), left to right."""
    if c < 0:
        return [~c]
    return _inorder_leaves(t, int(t.left[c])) + _inorder_leaves(t, int(t.right[c]))


@dataclass
class _BitTables:
    dtype: type
    full: int
    values: np.ndarray                      # (trees, max leaves): leaf value by left-to-right position
    features: dict                          # feature -> (sorted thresholds, prefix-AND table incl. NaN row)


class TreeEnsemble:
    def __init__(self, trees: list[_Tree], num_features: int, objective: str, sha256: str):
        self.trees, self.num_features, self.objective, self.sha256 = trees, num_features, objective, sha256
        self._bits: _BitTables | None | bool = None

    def prepare(self) -> TreeEnsemble:
        """Build the evaluation tables now (at model load) instead of on the first prediction."""
        self._bit_tables()
        return self

    def _bit_tables(self) -> _BitTables | None:
        if self._bits is not None:
            return self._bits or None
        n_leaves = max(len(t.leaf_value) for t in self.trees) if self.trees else 0
        if not self.trees or n_leaves > 64 or any((t.missing_type == 1).any() for t in self.trees if t.depth):
            self._bits = False
            return None
        dt = np.uint32 if n_leaves <= 32 else np.uint64
        full = int(np.iinfo(dt).max)
        values = np.zeros((len(self.trees), n_leaves), dtype=np.float64)
        rows: list[tuple[int, int, float, int, bool]] = []      # tree, feature, threshold, mask, NaN goes right
        for ti, t in enumerate(self.trees):
            if t.depth == 0:
                values[ti, 0] = t.leaf_value[0]
                continue
            order = _inorder_leaves(t, 0)
            pos = {leaf: p for p, leaf in enumerate(order)}
            values[ti, : len(order)] = t.leaf_value[order]
            for n in range(len(t.split_feature)):
                cleared = 0
                for leaf in _inorder_leaves(t, int(t.left[n])):
                    cleared |= 1 << pos[leaf]
                mt, thr = int(t.missing_type[n]), float(t.threshold[n])
                nan_right = (mt == 2 and not bool(t.default_left[n])) or (mt == 0 and not 0.0 <= thr)
                rows.append((ti, int(t.split_feature[n]), thr, full & ~cleared, nan_right))
        features = {}
        if rows:
            tree_of = np.array([r[0] for r in rows], dtype=np.int64)
            feat = np.array([r[1] for r in rows], dtype=np.int64)
            thr = np.array([r[2] for r in rows], dtype=np.float64)
            mask = np.array([r[3] for r in rows], dtype=dt)
            nan_right = np.array([r[4] for r in rows], dtype=bool)
            for f in np.unique(feat).tolist():
                sel = np.flatnonzero(feat == f)
                sel = sel[np.argsort(thr[sel], kind="stable")]
                k = len(sel)
                table = np.full((k + 2, len(self.trees)), full, dtype=dt)
                table[np.arange(1, k + 1), tree_of[sel]] = mask[sel]
                table[: k + 1] = np.bitwise_and.accumulate(table[: k + 1], axis=0)
                nr = sel[nan_right[sel]]
                np.bitwise_and.at(table[k + 1], tree_of[nr], mask[nr])
                features[f] = (thr[sel], table)
        self._bits = _BitTables(dt, full, values, features)
        return self._bits

    @classmethod
    def from_file(cls, path: str | Path) -> TreeEnsemble:
        raw = Path(path).read_bytes()
        return cls.from_text(raw.decode("utf-8"), hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_text(cls, text: str, sha256: str = "") -> TreeEnsemble:
        lines = text.splitlines()
        header: dict[str, str] = {}
        trees: list[_Tree] = []
        i = 0
        while i < len(lines) and not lines[i].startswith("Tree="):
            if "=" in lines[i]:
                k, v = lines[i].split("=", 1)
                header[k] = v
            i += 1
        while i < len(lines):
            line = lines[i]
            if line.startswith("end of trees"):
                break
            if not line.startswith("Tree="):
                i += 1
                continue
            block: dict[str, str] = {}
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("Tree="):
                if "=" in lines[i]:
                    k, v = lines[i].split("=", 1)
                    block[k] = v
                i += 1
            trees.append(cls._tree(block))
        nf = int(header.get("max_feature_idx", "-1")) + 1
        return cls(trees, nf, header.get("objective", ""), sha256)

    @staticmethod
    def _tree(block: dict[str, str]) -> _Tree:
        def arr(key, dtype):
            v = block.get(key, "").strip()
            return np.array(v.split(" "), dtype=dtype) if v else np.zeros(0, dtype=dtype)

        if int(block.get("num_cat", "0")) > 0:
            raise ValueError("categorical splits are not supported")
        leaf = arr("leaf_value", np.float64)
        if int(block.get("num_leaves", "1")) <= 1:
            z = np.zeros(0, dtype=np.int64)
            return _Tree(z, np.zeros(0), np.zeros(0, dtype=bool), z, z, z, leaf, 0)
        dt = arr("decision_type", np.int64)
        if (dt & 1).any():
            raise ValueError("categorical splits are not supported")
        left, right = arr("left_child", np.int64), arr("right_child", np.int64)
        return _Tree(arr("split_feature", np.int64), arr("threshold", np.float64), (dt & 2) > 0, (dt >> 2) & 3,
                     left, right, leaf, _depth(left, right))

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        bt = self._bit_tables()
        if bt is None or X.shape[0] == 0:
            return self.predict_levelwise(X)
        n, n_trees = X.shape[0], len(self.trees)
        acc = np.full((n, n_trees), bt.full, dtype=bt.dtype)
        for f, (thr, table) in bt.features.items():
            v = X[:, f]
            b = np.searchsorted(thr, v, side="left")            # number of thresholds below v = right-going nodes
            b[np.isnan(v)] = len(thr) + 1
            np.bitwise_and(acc, table[b], out=acc)
        lowest = acc & (~acc + bt.dtype(1))                     # the exit leaf is the lowest bit still set
        pos = np.log2(lowest).astype(np.int64)
        leaf_values = bt.values[np.arange(n_trees)[None, :], pos]
        out = np.zeros(n, dtype=np.float64)
        for t in range(n_trees):                                # LightGBM's summation order
            out += leaf_values[:, t]
        return out

    def predict_levelwise(self, X: np.ndarray) -> np.ndarray:
        """Node-by-node evaluation (reference implementation and fallback)."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        rows = np.arange(n)
        for t in self.trees:
            if t.depth == 0:
                out += t.leaf_value[0]
                continue
            node = np.zeros(n, dtype=np.int64)
            for _ in range(t.depth):
                inner = node >= 0
                if not inner.any():
                    break
                nd = np.where(inner, node, 0)
                v = X[rows, t.split_feature[nd]]
                mt = t.missing_type[nd]
                nan = np.isnan(v)
                v = np.where(nan & (mt != 2), 0.0, v)
                missing = ((mt == 1) & (np.abs(v) <= K_ZERO_THRESHOLD)) | ((mt == 2) & np.isnan(v))
                go_left = np.where(missing, t.default_left[nd], v <= t.threshold[nd])
                node = np.where(inner, np.where(go_left, t.left[nd], t.right[nd]), node)
            out += t.leaf_value[~node]
        return out
