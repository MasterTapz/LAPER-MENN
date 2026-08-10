"""Smoke test for the Layer 1 detection entry point (`detect_centres.py`, via
`oil_palm.demo.core.detect_image`) on two small committed sample tiles.

Not a metrics test — F1/precision/recall are measured properly by
`oil_palm/layer1/centre_eval_folds.py`, which needs the full ds_B dataset. This
just proves the detector loads, runs end-to-end on a real image, and doesn't
trip the `assert_classes()` class-map guard — the fastest possible check that
the weights + code are wired together correctly after the restructuring.
"""
import os

import pytest

from conftest import skip_unless

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SAMPLE_TILES = [
    os.path.join(FIXTURES, "sample_tile_1.jpg"),
    os.path.join(FIXTURES, "sample_tile_2.jpg"),
]


@pytest.mark.parametrize("tile", SAMPLE_TILES)
def test_detect_image_runs(has_layer1_weights, tile):
    skip_unless(has_layer1_weights, "Layer 1 weights not found — run scripts/download_assets.py first")
    assert os.path.isfile(tile), f"missing test fixture: {tile}"

    from oil_palm.demo import core

    df, info = core.detect_image(tile)
    # A near-empty tile can legitimately produce zero detections above the
    # conf=0.75 threshold — what this test guards is that the call completes
    # without raising (e.g. the assert_classes() Healthy/Unhealthy class-order
    # guard), not that it finds anything in particular. Column set differs
    # slightly between the zero-detection early-return and the populated case
    # (the latter adds a soft "unh" score column) — check the columns every
    # downstream consumer (demo_core, api.py) actually relies on.
    for col in ("cx", "cy", "conf", "cls", "deg"):
        assert col in df.columns
    assert "ok_n" in info and "n" in info
    assert info["n"] == len(df)
