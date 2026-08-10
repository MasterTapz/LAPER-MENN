"""Central path resolution for the SawitGuard-GNN production package.

Every path in this package is resolved relative to this repository checkout by
default, matching the exact subdirectory names the `layer1/`, `layer2/`, and
`data_clean/` scripts already expect on disk (`layer1/yolo12_runs/...`,
`layer2/risk_ranked.csv`, `data_clean/*.csv`, ...). `scripts/download_assets.py`
populates those same paths, either from a local checkout of
`oil-palm-detection-model-data/` or from the project's Hugging Face repos, so no
other module needs to know where the data actually came from.

Override with the environment variable below to point at a different location
instead (e.g. a shared network checkout of `oil-palm-detection-model-data/`)
without moving or re-downloading anything:

    SAWITGUARD_DATA_ROOT      -> replaces DATA_CLEAN_DIR

Note on weights: `y12.py`, `detect_centres.py`, `anom.py`, and the Layer 2
checkpoint loaders all resolve their weight paths relative to their OWN file
location (`layer1/yolo12_runs/...`, `layer2/stgnn_v3_photo.pt`, ...) rather
than through this module — that is intentional (it is what lets those files
also run standalone / from a notebook unmodified). WEIGHTS_ROOT below is
therefore NOT a free-standing override: it is simply PKG_ROOT, documented
here so `scripts/download_assets.py` has one obvious place to write weights
into that both it and the layer1/layer2 modules agree on. Do not add a
SAWITGUARD_WEIGHTS_ROOT env override without also updating those modules —
otherwise assets land in one place while the code still looks in another.
"""
import os

_THIS_FILE = os.path.abspath(__file__)
PKG_ROOT = os.path.dirname(_THIS_FILE)                 # .../src/oil_palm
SRC_ROOT = os.path.dirname(PKG_ROOT)                   # .../src
REPO_ROOT = os.path.dirname(SRC_ROOT)                  # repository root

# Code packages (always repo-relative — these never move).
LAYER1_DIR = os.path.join(PKG_ROOT, "layer1")
LAYER2_DIR = os.path.join(PKG_ROOT, "layer2")
DEMO_DIR = os.path.join(PKG_ROOT, "demo")

# Data (overridable — this is what download_assets.py populates by default).
DATA_CLEAN_DIR = os.environ.get(
    "SAWITGUARD_DATA_ROOT", os.path.join(REPO_ROOT, "data_clean")
)

# Weights (NOT independently overridable — see note above). Always PKG_ROOT,
# so `<WEIGHTS_ROOT>/layer1/yolo12_runs/...` == what y12.py/detect_centres.py
# already resolve on their own.
WEIGHTS_ROOT = PKG_ROOT

# Non-package repo-level directories.
WEB_DIR = os.path.join(REPO_ROOT, "web")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
FIGURES_DIR = os.path.join(DOCS_DIR, "figures")


def ensure_on_path():
    """Put layer1/ and layer2/ on sys.path so their flat, __file__-relative
    `import y12`, `import detect_centres`, `import models_real` style imports
    keep working unmodified inside this package."""
    import sys
    for p in (LAYER1_DIR, LAYER2_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
