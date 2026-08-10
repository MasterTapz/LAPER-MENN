#!/usr/bin/env python
"""Thin CLI wrapper around the Layer 2 (Eg9PP) training scripts.

    python scripts/train_layer2.py                # stgnn_v3_photo.pt — the demo checkpoint
    python scripts/train_layer2.py --full-model    # also stgnn_final.pt — the 24-feature full model

Needs data_clean/layer2_{nodes,panel,edges}.csv on disk (included in every
`download_assets.py` run). Runs on CPU (Layer 2 pins DEVICE=cpu on purpose —
1,200 nodes x 45 censuses does not benefit enough from a GPU to be worth the
transfer overhead). Runtime: a few minutes for the photo variant; the full
decomposition in `run_real.py`/`run_v2.py` (not run by this script — those are
the *evaluation* drivers, this script only trains the two final checkpoints)
is documented as ~22 min in docs/RESULTS.md.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER2_DIR = os.path.join(REPO_ROOT, "src", "oil_palm", "layer2")


def run(script):
    print(f"\n=== {script} ===")
    r = subprocess.run([sys.executable, script], cwd=LAYER2_DIR)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-model", action="store_true",
                     help="also train stgnn_final.pt (24-feature full model, needs a 2-visit history — not what the demo uses)")
    a = ap.parse_args()

    run("train_final_v3.py")
    if a.full_model:
        run("train_final.py")


if __name__ == "__main__":
    main()
