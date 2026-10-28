"""
Run a config from `configs/` on a Mac (CPU or Apple MPS) without modifying the original code.

run_train_test.run() hard-codes CUDA (device=f'cuda:{id}', parallelize=True, GPU discovery via
nvidia-smi). This launcher wraps the functions it calls so that training runs on a single local
device, then calls run() unchanged.

Usage (from the repo root):
    .venv/bin/python local/run_local.py -c configs/ouffn-full.yaml
    .venv/bin/python local/run_local.py -c configs/ouffn-full.yaml --device mps --num-epochs 10 --factor-model PCA:5
"""
import argparse
import contextlib
import functools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # run_train_test uses paths relative to the repo root

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import torch
import yaml

import run_train_test
import train_test

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FF5_PATH = "data/F-F_Research_Data_5_Factors_2x3_daily.CSV"


def prepare_dates_file():
    """
    run() only uses this file for its trading dates. Its original parser (header=2) no longer
    matches the current file layout (longer preamble, trailing copyright line), so write the
    file run() expects: date index (YYYYMMDD) and factor columns.
    """
    if os.path.exists(FF5_PATH):
        return
    os.makedirs("data", exist_ok=True)
    ff5 = pd.read_csv(FF5_URL, skiprows=3, index_col=0)
    ff5 = ff5[pd.to_numeric(ff5.index, errors="coerce").notna()]
    ff5.index = ff5.index.astype(int)
    ff5.to_csv(FF5_PATH)


def local_device(func, device):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.update(device=device, parallelize=False, device_ids=None)
        return func(*args, **kwargs)
    return wrapper


def patch_for_device(device, detect_anomaly):
    run_train_test.get_free_gpu_ids = lambda *a, **k: [0]
    run_train_test.test = local_device(train_test.test, device)
    run_train_test.estimate = local_device(train_test.estimate, device)
    if device.startswith("mps"):
        # train_test.test() calls torch.cuda.empty_cache() for every non-CPU device
        torch.cuda.device = lambda *a, **k: contextlib.nullcontext()
        torch.cuda.empty_cache = torch.mps.empty_cache
    # The original modules enable anomaly detection at import time, which slows down backward()
    torch.autograd.set_detect_anomaly(detect_anomaly)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="path to a .yaml config, e.g. configs/ouffn-full.yaml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--num-epochs", type=int, help="override config num_epochs")
    parser.add_argument("--factor-model", action="append", metavar="NAME:N",
                        help="only test these residuals, e.g. PCA:5 (repeatable); default: all from config")
    parser.add_argument("--results-tag", help="override config results_tag")
    parser.add_argument("--detect-anomaly", action="store_true",
                        help="keep torch anomaly detection on, as in the original code (slower)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.results_tag is not None:
        config["results_tag"] = args.results_tag
    if args.factor_model:
        selected = {name: [] for name in config["factor_models"]}
        for item in args.factor_model:
            name, n = item.split(":")
            selected[name].append(int(n))
        config["factor_models"] = selected
    if config.get("use_residual_weights"):
        raise SystemExit("use_residual_weights=True needs the residual composition matrices, which are not in this repo")

    prepare_dates_file()
    patch_for_device(args.device, args.detect_anomaly)
    run_train_test.run(config)


if __name__ == "__main__":
    main()
