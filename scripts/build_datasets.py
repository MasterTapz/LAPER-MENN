#!/usr/bin/env python
"""Rebuild the frozen data_clean/*.csv files from raw sources.

    python scripts/build_datasets.py            # both layers
    python scripts/build_datasets.py --layer1    # only layer1_crowns.csv etc.
    python scripts/build_datasets.py --layer2    # only layer2_nodes/panel/edges.csv

Requires raw sources on disk first: ds_B/ (Layer 1, via
`python scripts/download_assets.py --full`) and data_clean/Eg9PP_Phenotypes.csv
(Layer 2, included in every download mode). Both builders hard-assert their
headline counts (5,077 unique trees / 66 positives; 1,200 nodes / 45 censuses /
3,354 edges) and abort loudly if the numbers drift — that is intentional, not
a bug to work around.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN = os.path.join(REPO_ROOT, "data_clean")


def run(script):
    print(f"\n=== {script} ===")
    r = subprocess.run([sys.executable, script], cwd=DATA_CLEAN)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer1", action="store_true")
    ap.add_argument("--layer2", action="store_true")
    a = ap.parse_args()
    both = not (a.layer1 or a.layer2)

    if a.layer1 or both:
        run("build_layer1.py")
    if a.layer2 or both:
        run("build_layer2_real.py")


if __name__ == "__main__":
    main()
