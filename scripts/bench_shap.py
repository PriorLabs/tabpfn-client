"""Compare per-row shap.PermutationExplainer with batched permutation_shap.

Arms:
  A  shap.PermutationExplainer, fit_mode="fit_with_cache" (one request per row)
  B  shap.PermutationExplainer, default fit_mode (one request per row)
  C  permutation_shap, default fit_mode (one request)
  D  permutation_shap, fit_mode="fit_with_cache" (one request)

Run against a server (TABPFN_TOKEN or a cached login; TABPFN_CLIENT_API_URL
for a dev deployment):

    uv run --with shap python scripts/bench_shap.py --n-explain 20

`--local` swaps TabPFN for a scikit-learn model, to check the arms agree
without a server. Extrapolate to more explained rows from two `--n-explain`
values: per-row arms scale with requests, batched arms with masked rows.
"""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Any

import numpy as np
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

from tabpfn_client import TabPFNClassifier, set_access_token
from tabpfn_client.explain import permutation_shap


class Counted:
    """Wrap a model's predict_proba to count calls, rows and server seconds."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls = 0
        self.rows = 0
        self.server: dict[str, float] = {}

    def __call__(self, X: Any) -> np.ndarray:
        self.calls += 1
        self.rows += len(X)
        proba = self.model.predict_proba(X)
        timings = getattr(self.model, "last_predict_timings", None) or {}
        for key, seconds in timings.items():
            if isinstance(seconds, (int, float)):
                self.server[key] = self.server.get(key, 0.0) + seconds
        return proba


def make_model(local: bool, cache: bool, args: argparse.Namespace) -> Any:
    if local:
        return HistGradientBoostingClassifier(random_state=0)
    return TabPFNClassifier.create_default_for_version(
        "v3.5",
        fit_mode="fit_with_cache" if cache else None,
        n_estimators=args.n_estimators,
        random_state=0,
    )


def run_arm(name: str, batched: bool, cache: bool, data: dict, args) -> dict:
    model = make_model(args.local, cache, args)
    start = perf_counter()
    model.fit(data["X_train"], data["y_train"])
    fit_s = perf_counter() - start

    predict = Counted(model)
    start = perf_counter()
    if batched:
        result = permutation_shap(
            predict,
            data["X_explain"],
            data["background"],
            n_permutations=args.n_permutations,
            # Predicts against a cached fit are capped per request.
            batch_size=args.cache_batch_size if cache else None,
            random_state=0,
        )
        values = result.values[..., 1]
    else:
        n_features = data["X_explain"].shape[1]
        explainer = shap.PermutationExplainer(
            lambda X: predict(X)[:, 1], data["background"], seed=0
        )
        explanation = explainer(
            data["X_explain"],
            max_evals=args.n_permutations * (2 * n_features + 1),
            silent=True,
        )
        values = explanation.values  # pyright: ignore[reportAttributeAccessIssue]
    shap_s = perf_counter() - start

    return {
        "arm": name,
        "fit_s": round(fit_s, 2),
        "shap_s": round(shap_s, 2),
        "requests": predict.calls,
        "rows_predicted": predict.rows,
        "server_s": {k: round(v, 2) for k, v in predict.server.items()},
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark per-row and batched SHAP.")
    parser.add_argument("--n-train", type=int, default=1_000)
    parser.add_argument("--n-features", type=int, default=13)
    parser.add_argument("--n-explain", type=int, default=20)
    parser.add_argument("--n-background", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--cache-batch-size", type=int, default=10_000)
    parser.add_argument("--arms", default="ABCD")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    if not args.local and "TABPFN_TOKEN" in os.environ:
        set_access_token(os.environ["TABPFN_TOKEN"])

    X, y = make_classification(
        n_samples=args.n_train + args.n_explain,
        n_features=args.n_features,
        n_informative=min(8, args.n_features),
        n_redundant=min(3, max(0, args.n_features - 8)),
        random_state=42,
    )
    X_train, y_train = X[: args.n_train], y[: args.n_train]
    rng = np.random.default_rng(42)
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_explain": X[args.n_train :],
        "background": X_train[
            rng.choice(args.n_train, args.n_background, replace=False)
        ],
    }

    arms = {
        "A": (False, True),
        "B": (False, False),
        "C": (True, False),
        "D": (True, True),
    }
    results = []
    for name in args.arms:
        batched, cache = arms[name]
        result = run_arm(name, batched, cache, data, args)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "values"}))

    # Different arms sample different orderings, so values agree only roughly;
    # more permutations tighten it.
    reference = results[0]
    for result in results[1:]:
        a, b = reference["values"].ravel(), result["values"].ravel()
        print(
            json.dumps(
                {
                    "compare": f"{reference['arm']} vs {result['arm']}",
                    "corr": round(float(np.corrcoef(a, b)[0, 1]), 3),
                    "max_abs_diff": round(float(np.abs(a - b).max()), 4),
                }
            )
        )


if __name__ == "__main__":
    main()
