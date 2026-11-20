#!/usr/bin/env python3
"""Convenience CLI entrypoint for datasetcleaner-reporter.

Can be run directly via:
    python reporter.py --dir ./output
    python reporter.py --interval 60   # daemon mode every 60 seconds
"""

import sys
from pathlib import Path

# Ensure dataset_cleaner is importable from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_cleaner.reporter import main

if __name__ == "__main__":
    sys.exit(main())
