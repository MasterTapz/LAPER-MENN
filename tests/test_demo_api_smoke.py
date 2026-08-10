"""Smoke test for the Starlette demo API — proves the HTTP layer, the Layer 1
detector, and the Layer 2 risk-ranking checkpoint are all wired together
correctly after the restructuring. Not a metrics test.
"""
import pytest

from conftest import skip_unless


@pytest.fixture(scope="module")
def client(has_layer1_weights, has_layer2_weights, has_ds_b_images):
    skip_unless(
        has_layer1_weights and has_layer2_weights and has_ds_b_images,
        "weights/raw ds_B images not found — run `scripts/download_assets.py --full` first "
        "(the API's /api/samples reads raw ds_B tiles directly, not just weights)",
    )
    from starlette.testclient import TestClient
    from oil_palm.demo.api import app

    return TestClient(app)


def test_samples_endpoint(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    body = r.json()
    assert "samples" in body and "facts" in body
    assert len(body["samples"]) > 0
    assert "thumb" in body["samples"][0]


def test_analyze_endpoint(client):
    r = client.post("/api/analyze", data={"sample": "0"})
    assert r.status_code == 200
    body = r.json()
    for key in ("detect", "risk", "risk_soft", "edges", "crowns", "readiness", "checks"):
        assert key in body


def test_eg9pp_endpoint(client):
    r = client.get("/api/eg9pp")
    assert r.status_code == 200
