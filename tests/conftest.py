"""Shared pytest fixtures.

Most of this test suite needs weights/data that are NOT committed to git (see
.gitignore + scripts/download_assets.py) — they have to be downloaded or
symlinked in first. Rather than hard-failing on a fresh clone, tests that need
them are skipped with a clear reason until `download_assets.py` has been run.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from oil_palm.config import DATA_CLEAN_DIR, LAYER1_DIR, LAYER2_DIR  # noqa: E402


def _exists(*parts):
    return os.path.isfile(os.path.join(*parts))


@pytest.fixture(scope="session")
def has_layer2_data():
    return _exists(DATA_CLEAN_DIR, "layer2_nodes.csv") and _exists(DATA_CLEAN_DIR, "layer2_panel.csv")


@pytest.fixture(scope="session")
def has_layer1_weights():
    return _exists(LAYER1_DIR, "yolo12_runs", "yolo12n_base_1fold_fold0_s42", "weights", "best.pt")


@pytest.fixture(scope="session")
def has_layer2_weights():
    return _exists(LAYER2_DIR, "stgnn_v3_photo.pt") and _exists(LAYER2_DIR, "risk_ranked.csv")


@pytest.fixture(scope="session")
def has_ds_b_images():
    # api.samples()/core.sample_images() reads raw ds_B tiles directly — these
    # are only pulled by `download_assets.py --full`, not the demo-only default.
    train_dir = os.path.join(LAYER1_DIR, "ds_B", "train")
    return os.path.isdir(train_dir) and any(f.endswith(".jpg") for f in os.listdir(train_dir))


def skip_unless(cond, reason):
    if not cond:
        pytest.skip(reason)
