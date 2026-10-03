# Benchmark provenance

The published 1,000-query artifact is a deterministic synthetic regression
benchmark. During development, an exploratory oracle was run first. That
unpublished pilot used weights genre=0.50, theme=0.25, demographic=0.10,
length=0.15, noise standard deviation 0.12, and pool sizes 12/40/200 for
easy/medium/hard. Its D coverage@5 was 9.64% and binary hit_rate@5 was 37.00%.

After observing that pilot, the oracle was calibrated for a more useful
controlled stress case: genre=0.50, theme=0.30, demographic=0.10, length=0.10,
noise standard deviation 0.05, pool sizes 8/20/60, and softmax temperature
0.035. The final run then generated all 1,000 labels and rankings with seed
20261003. The final result is therefore reproducible for the current protocol,
but it is not a pre-registered, independent accuracy estimate.

Target selection was fixed by SHA-256 ordering and did not use final
predictions. The targets were not removed from the existing weak-supervision
training endpoints; `testset_manifest.json` records the overlap count. This
benchmark is suitable for regression and method comparisons under its declared
oracle, not for claims about real users or MAL recommendations.
