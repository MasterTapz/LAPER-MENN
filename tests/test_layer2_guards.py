"""pytest entry point for the Layer 2 leakage-guard suite.

The ~90 assertions / 4 leakage guards in `oil_palm/layer2/test_dataset.py` are
intentionally left untouched (see that file's own docstring) — this just makes
them discoverable and runnable via `pytest`, by running the script as a
subprocess and checking its exit code, the same as the documented standalone
invocation `python src/oil_palm/layer2/test_dataset.py`.
"""
import os
import subprocess
import sys

import pytest

from conftest import LAYER2_DIR, skip_unless


def test_layer2_dataset_guards_pass(has_layer2_data):
    skip_unless(has_layer2_data, "data_clean/layer2_*.csv not found — run scripts/download_assets.py first")

    script = os.path.join(LAYER2_DIR, "test_dataset.py")
    result = subprocess.run(
        [sys.executable, script],
        cwd=LAYER2_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"layer2 guard suite failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
