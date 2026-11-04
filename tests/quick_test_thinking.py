"""
Thinking Mode Smoke Test
------------------------
Toy script to check that thinking mode works against a live server (prod by default).

For each task it fits once without thinking (baseline) and once per thinking effort,
then prints fit time, server fit timings, test score and how far the predictions moved
from the baseline. Thinking fits that finish as fast as the baseline and return the
same predictions are flagged: that is what "thinking does nothing" looks like.

Run (prod):
    unset TABPFN_CLIENT_API_URL
    export TABPFN_TOKEN=<prod token>
    python tests/quick_test_thinking.py                       # mini + medium, both tasks
    python tests/quick_test_thinking.py --efforts micro mini low medium high
    python tests/quick_test_thinking.py --task clf --efforts medium

Thinking fits use the separate, daily-limited thinking budget; each effort is one fit
per task.
"""

import argparse
import importlib.metadata
import logging
import time
from typing import cast
from unittest.mock import patch

import httpx
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn_client import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.client import ServiceClient
from tabpfn_client.service_wrapper import UserAuthenticationClient

logging.basicConfig(level=logging.INFO)

EFFORTS = ["micro", "mini", "low", "medium", "high", "xhigh", "max"]
TRAIN_ROWS = 300


def print_target() -> None:
    """Print which server and account this run hits, before spending any budget."""
    try:
        version = importlib.metadata.version("tabpfn-client")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    base_url = str(ServiceClient.base_url)
    print(f"tabpfn-client {version}, server {base_url}")
    token = UserAuthenticationClient.resolve_token()
    if not token:
        print("no token found (set TABPFN_TOKEN)")
        return
    try:
        r = httpx.get(
            f"{base_url.rstrip('/')}/user/profile",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        r.raise_for_status()
        print(f"account {r.json().get('email')}")
    except Exception as e:  # informational only
        print(f"could not read account profile: {e!r}")


def run(make_estimator, X_train, y_train, X_test, y_test, score, efforts, timeout_s):
    baseline = None
    rows = []
    for effort in [None, *efforts]:
        name = effort or "standard"
        kwargs = (
            {}
            if effort is None
            else {
                "thinking_effort": effort,
                "thinking_timeout_s": timeout_s,
            }
        )
        print(f"\n=== {name} ===")
        est = make_estimator(**kwargs)
        try:
            t0 = time.monotonic()
            est.fit(X_train, y_train)
            fit_s = time.monotonic() - t0
            t0 = time.monotonic()
            pred, metrics = score(est, X_test, y_test)
            predict_s = time.monotonic() - t0
        except Exception as e:
            print(f"{name} FAILED: {e!r}")
            rows.append((name, None, None, None, f"FAILED: {type(e).__name__}"))
            continue
        print(f"model_id: {getattr(est, 'model_id_', None)}")
        print(f"fit_timings: {getattr(est, 'fit_timings_', None)}")
        print(f"last meta: {est._last_meta}")

        if baseline is None:
            baseline = (pred, fit_s)
            note = ""
        else:
            diff = float(np.abs(pred - baseline[0]).max())
            note = f"max |Δpred| vs standard {diff:.2e}"
            if diff < 1e-6:
                note += "  <-- SAME AS STANDARD"
            if fit_s < 1.5 * baseline[1]:
                note += "  <-- fit not slower than standard"
        rows.append((name, fit_s, predict_s, metrics, note))

    print("\n--- summary ---")
    for name, fit_s, predict_s, metrics, note in rows:
        if fit_s is None:
            print(f"{name:>9}  {note}")
            continue
        m = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{name:>9}  fit {fit_s:7.1f}s  predict {predict_s:5.1f}s  {m}  {note}")


def score_clf(est, X_test, y_test):
    proba = est.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    return proba, {
        "acc": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba[:, 1]),
    }


def score_reg(est, X_test, y_test):
    pred = cast(np.ndarray, est.predict(X_test))
    return pred, {"r2": r2_score(y_test, pred)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["clf", "reg", "both"], default="both")
    parser.add_argument(
        "--efforts", nargs="+", choices=EFFORTS, default=["mini", "medium"]
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="thinking_timeout_s per fit (server caps at 2400)",
    )
    args = parser.parse_args()

    # Patch webbrowser.open to prevent browser login
    with patch("webbrowser.open", return_value=False):
        print_target()

        if args.task in ("clf", "both"):
            X, y = load_breast_cancer(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )
            print("\n##### classification (breast cancer) #####")
            run(
                lambda **kw: TabPFNClassifier(random_state=42, **kw),
                X_train[:TRAIN_ROWS],
                y_train[:TRAIN_ROWS],
                X_test,
                y_test,
                score_clf,
                args.efforts,
                args.timeout,
            )

        if args.task in ("reg", "both"):
            X, y = load_diabetes(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )
            print("\n##### regression (diabetes) #####")
            run(
                lambda **kw: TabPFNRegressor(random_state=42, **kw),
                X_train[:TRAIN_ROWS],
                y_train[:TRAIN_ROWS],
                X_test,
                y_test,
                score_reg,
                args.efforts,
                args.timeout,
            )
