"""`python -m algorithm2_benchmark <stage> --run <run_id> [--level fixture|smoke|main]` (see README.md)."""

import sys

from algorithm2_benchmark.pipeline.cli import main

sys.exit(main())
