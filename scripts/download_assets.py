#!/usr/bin/env python
"""Populate weights + data from either a local `oil-palm-detection-model-data/`
checkout or the project's Hugging Face repos.

    python scripts/download_assets.py --from-local ../oil-palm-detection-model-data
    python scripts/download_assets.py --from-local ../oil-palm-detection-model-data --full
    python scripts/download_assets.py                      # pulls from Hugging Face (demo-only)
    python scripts/download_assets.py --full                # pulls everything from Hugging Face

Every path this script writes to is exactly what `src/oil_palm/layer1/y12.py`,
`detect_centres.py`, `anom.py`, `layer2/*`, `demo/core.py`, and `data_clean/build_*.py`
already resolve on their own (see `src/oil_palm/config.py`) — nothing downstream
needs to change based on where the assets came from.

--demo-only (default): the small subset needed to run the web demo — frozen CSVs,
the single-photo Layer 2 checkpoint, the one YOLO run the demo loads first, and
the small Layer 1 result JSONs. ~65 MB.

--full: everything, including all 4 YOLO fold runs, the Peru cross-site weights
+ raw images, the full Layer 2 checkpoint, and the raw ds_B UAV tiles needed to
rebuild the frozen CSVs from scratch. ~520 MB.
"""
import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from oil_palm.config import DATA_CLEAN_DIR, LAYER1_DIR, LAYER2_DIR, PKG_ROOT  # noqa: E402

PERU_DIR_NAME = "Oil Palm Tree Detection 4.v15i.tensorflow"  # matches anom.py's own hardcoded name


def _copy_file(src, dst):
    if not os.path.isfile(src):
        print(f"  SKIP (not found): {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {os.path.relpath(dst, REPO_ROOT)}")
    return True


def _copy_tree(src, dst):
    if not os.path.isdir(src):
        print(f"  SKIP (not found): {src}")
        return False
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    n = sum(len(f) for _, _, f in os.walk(dst))
    print(f"  {os.path.relpath(dst, REPO_ROOT)}  ({n} files)")
    return True


def demo_only_map(md):
    """md = path to a local oil-palm-detection-model-data/ checkout."""
    files = [
        (f"{md}/datasets/layer1_uav_crowns/frozen/layer1_crowns.csv", f"{DATA_CLEAN_DIR}/layer1_crowns.csv"),
        (f"{md}/datasets/layer1_uav_crowns/frozen/layer1_tiles_disjoint.csv", f"{DATA_CLEAN_DIR}/layer1_tiles_disjoint.csv"),
        (f"{md}/datasets/layer2_eg9pp_panel/frozen/layer2_nodes.csv", f"{DATA_CLEAN_DIR}/layer2_nodes.csv"),
        (f"{md}/datasets/layer2_eg9pp_panel/frozen/layer2_panel.csv", f"{DATA_CLEAN_DIR}/layer2_panel.csv"),
        (f"{md}/datasets/layer2_eg9pp_panel/frozen/layer2_edges.csv", f"{DATA_CLEAN_DIR}/layer2_edges.csv"),
        (f"{md}/datasets/layer2_eg9pp_panel/Eg9PP_Phenotypes.csv", f"{DATA_CLEAN_DIR}/Eg9PP_Phenotypes.csv"),
        (f"{md}/weights/layer2/stgnn_v3_photo.pt", f"{LAYER2_DIR}/stgnn_v3_photo.pt"),
        (f"{md}/weights/layer2/risk_ranked.csv", f"{LAYER2_DIR}/risk_ranked.csv"),
        (f"{md}/weights/layer2/risk_ranked.meta.json", f"{LAYER2_DIR}/risk_ranked.meta.json"),
        (f"{md}/weights/layer1/yolo12_runs/yolo12n_base_1fold_fold0_s42/weights/best.pt",
         f"{LAYER1_DIR}/yolo12_runs/yolo12n_base_1fold_fold0_s42/weights/best.pt"),
    ]
    trees = [
        (f"{md}/weights/layer1/yolo12_results", f"{LAYER1_DIR}/yolo12_results"),
    ]
    return files, trees


def full_extra_map(md):
    """Additional assets pulled only with --full, on top of demo_only_map()."""
    files = [
        (f"{md}/weights/layer2/stgnn_final.pt", f"{LAYER2_DIR}/stgnn_final.pt"),
        (f"{md}/weights/layer1/anom_peru/stage1_model.pkl", f"{LAYER1_DIR}/stage1_model.pkl"),
        (f"{md}/weights/layer1/anom_peru/stage1_summary.json", f"{LAYER1_DIR}/stage1_summary.json"),
    ]
    trees = [
        (f"{md}/weights/layer1/yolo12_runs", f"{LAYER1_DIR}/yolo12_runs"),  # overwrites/completes the 3 remaining folds
        (f"{md}/weights/layer1/anom_peru/stage1_fold0_s42", f"{LAYER1_DIR}/anom_runs/stage1_fold0_s42"),
        (f"{md}/weights/layer1/lightgbm_health", f"{LAYER1_DIR}/lightgbm_health"),  # may not exist yet — see note below
        (f"{md}/datasets/layer1_uav_crowns/ds_B", f"{LAYER1_DIR}/ds_B"),
        (f"{md}/datasets/peru_palm_anomaly", os.path.join(PKG_ROOT, PERU_DIR_NAME)),
    ]
    return files, trees


def from_local(model_data_dir, full):
    print(f"Copying from local checkout: {model_data_dir}")
    files, trees = demo_only_map(model_data_dir)
    if full:
        f2, t2 = full_extra_map(model_data_dir)
        files += f2
        trees += t2
    for src, dst in files:
        _copy_file(src, dst)
    for src, dst in trees:
        _copy_tree(src, dst)


def from_huggingface(full):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed — pip install -r requirements-dev.txt, "
            "or use --from-local <path> if you already have oil-palm-detection-model-data/ on disk."
        )
    import yaml
    cfg_path = os.path.join(REPO_ROOT, "configs", "default.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))["huggingface"]
    if "REPLACE_ME" in cfg["model_repo"] or "REPLACE_ME" in cfg["dataset_repo"]:
        raise SystemExit(
            "configs/default.yaml still has placeholder Hugging Face repo ids. "
            "Fill in huggingface.model_repo / huggingface.dataset_repo once the "
            "repos are created, or use --from-local <path-to-model-data-folder>."
        )
    tmp_models = snapshot_download(repo_id=cfg["model_repo"], repo_type="model")
    tmp_data = snapshot_download(repo_id=cfg["dataset_repo"], repo_type="dataset")
    # snapshot_download gives us a local cache dir shaped exactly like the HF repo,
    # i.e. it already has the weights/ and datasets/ subfolders as uploaded — reuse
    # the same local-copy logic by pointing "model_data_dir" at a stitched view.
    staging = os.path.join(REPO_ROOT, ".hf_cache_view")
    os.makedirs(os.path.join(staging, "weights"), exist_ok=True)
    os.makedirs(os.path.join(staging, "datasets"), exist_ok=True)
    for name in os.listdir(tmp_models):
        s, d = os.path.join(tmp_models, name), os.path.join(staging, "weights", name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
    for name in os.listdir(tmp_data):
        s, d = os.path.join(tmp_data, name), os.path.join(staging, "datasets", name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
    from_local(staging, full)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-local", metavar="PATH", default=None,
                     help="path to a local oil-palm-detection-model-data/ checkout, instead of Hugging Face")
    ap.add_argument("--full", action="store_true",
                     help="download everything (~520MB) instead of just the demo subset (~65MB)")
    a = ap.parse_args()

    print(f"mode: {'full' if a.full else 'demo-only'}")
    if a.from_local:
        from_local(os.path.abspath(a.from_local), a.full)
    else:
        from_huggingface(a.full)
    print("\ndone. Note: src/oil_palm/layer1/lightgbm_health/ and the Peru raw-image "
          "folder are only populated in --full mode. The LightGBM crown-health model "
          "(model.txt + meta.json) WAS re-trained and is shipped — regenerate it any "
          "time with `python -m oil_palm.layer1.exp_health` (~23s CPU); it reproduces "
          "PR-AUC 0.182 +/- 0.059 leave-one-ortho-out. See docs/RESULTS.md.")


if __name__ == "__main__":
    main()
