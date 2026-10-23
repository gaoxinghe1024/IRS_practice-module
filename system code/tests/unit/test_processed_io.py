import numpy as np
import pyarrow.parquet as pq

from pipelines.processed_io import ProcessedWriter, mix_hash


def test_dedupe_keeps_first_occurrence_and_counts_conflicts(tmp_path):
    w = ProcessedWriter(tmp_path, n_partitions=2)
    w.append_ratings(np.array([1, 1, 2, 1]), np.array([5, 5, 7, 6]), np.array([9, 3, 8, 7]))
    w.append_ratings(np.array([2]), np.array([7]), np.array([8]))
    stats = w.close_ratings()
    assert stats["duplicate_rows_dropped"] == 2 and stats["conflicting_pairs"] == 1 and stats["rows"] == 3
    rows = {}
    for part in (tmp_path / "ratings").glob("*.parquet"):
        for r in pq.read_table(part).to_pylist():
            rows[(r["offline_user_id"], r["item_idx"])] = r["rating"]
    assert rows[(1, 5)] == 9          # first in file order, not "latest"


def test_mix_hash_is_deterministic_and_uniformish():
    ids = np.arange(100_000)
    a, b = mix_hash(ids, 7), mix_hash(ids, 7)
    assert np.array_equal(a, b) and 0.45 < (a < 0.5).mean() < 0.55
    assert not np.array_equal(a, mix_hash(ids, 8))
