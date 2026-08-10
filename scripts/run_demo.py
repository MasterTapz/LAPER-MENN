#!/usr/bin/env python
"""Launch the SawitGuard web demo (Starlette API + React frontend).

    python scripts/run_demo.py                  # http://localhost:8000
    python scripts/run_demo.py --port 8080

Requires weights + data to already be present — run
`python scripts/download_assets.py` first if you haven't.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    import uvicorn
    from oil_palm.demo.api import app, _banner_bobot

    print(f"Prediksi Pohon Berisiko  ->  http://{a.host}:{a.port}")
    print("  React + Babel di-vendor lokal; demo ini tidak butuh internet.")
    _banner_bobot()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
