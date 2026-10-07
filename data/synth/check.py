#!/usr/bin/env python
"""Fail unless a synthetic run's test accuracy reaches the target (smoke-test criterion)."""

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path, help="experiment.output_dir of the run")
parser.add_argument("--min-accuracy", type=float, default=0.9)
args = parser.parse_args()

accuracy = json.loads((args.run_dir / "final_metrics.json").read_text())["test"]["scan"]["accuracy"]
verdict = "PASS" if accuracy >= args.min_accuracy else "FAIL"
print(f"{verdict}: {args.run_dir} test accuracy {accuracy:.4f} (target {args.min_accuracy})")
sys.exit(0 if verdict == "PASS" else 1)
