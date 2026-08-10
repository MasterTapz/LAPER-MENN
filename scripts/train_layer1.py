#!/usr/bin/env python
"""Thin CLI wrapper around `oil_palm.layer1.train_folds_gpu` (YOLOv12n crown detector).

    python scripts/train_layer1.py                # default: fold0 and fold2
    python scripts/train_layer1.py --folds all     # all 3 folds (recommended for paper-comparable numbers)

Needs a CUDA GPU (cu126, not cu130 — see requirements.txt) and raw ds_B tiles
on disk (`python scripts/download_assets.py --full` then
`python scripts/build_datasets.py --layer1` if data_clean/layer1_crowns.csv
doesn't exist yet). Runtime: roughly 15-45 min per fold depending on GPU.

This wrapper does not contain training logic itself — see the note at the top
of train_folds_gpu.py about why it always passes all three fold names to the
underlying `y12.train_arm()` resume mechanism even when training fewer folds.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER1_DIR = os.path.join(REPO_ROOT, "src", "oil_palm", "layer1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=None, help='e.g. "all" or "fold0,fold2" — sets the FOLDS env var')
    ap.add_argument("--epochs", type=int, default=None, help="sets the EPOCHS env var (default 25)")
    a = ap.parse_args()

    env = os.environ.copy()
    if a.folds:
        env["FOLDS"] = a.folds
    if a.epochs:
        env["EPOCHS"] = str(a.epochs)

    r = subprocess.run([sys.executable, "train_folds_gpu.py"], cwd=LAYER1_DIR, env=env)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
