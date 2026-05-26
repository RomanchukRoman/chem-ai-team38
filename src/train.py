"""End-to-end training entrypoint.

Trains IC50, CC50 and (optionally) SI ensembles and dumps numpy artifacts
that `predict.py` reads to assemble a submission CSV.

Run:
    uv run python -m src.train --n-seeds 3 --n-splits 5
    uv run python -m src.train --quick   # smoke test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data import TARGETS, load_dataset, target_slug
from src.models import FoldConfig, _rmse, train_target
from src.seeds import SEED, set_seed

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ChemAI ensemble.")
    p.add_argument("--n-seeds", type=int, default=3, help="Number of seeds per target.")
    p.add_argument("--n-splits", type=int, default=5, help="KFold splits.")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["IC50, mM", "CC50, mM", "SI"],
        help="Subset of targets to train.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test config: 1 seed × 3 splits with only catboost.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    print(f"train={ds.X_train.shape}  test={ds.X_test.shape}")

    if args.quick:
        seeds = (args.seed,)
        n_splits = 3
        model_names = ("catboost",)
    else:
        seeds = tuple(args.seed + i * 35 for i in range(args.n_seeds))
        n_splits = args.n_splits
        model_names = ("catboost", "lightgbm", "xgboost")
    cfg = FoldConfig(n_splits=n_splits, seeds=seeds, preprocessor_seed=args.seed)

    report: dict[str, object] = {
        "config": {
            "seed": args.seed,
            "seeds": list(seeds),
            "n_splits": n_splits,
            "model_names": list(model_names),
            "targets": args.targets,
            "quick": args.quick,
        },
        "per_target": {},
    }

    np.save(args.artifacts / "test_index.npy", ds.test_index.to_numpy())
    np.save(args.artifacts / "y_train.npy", ds.y_train[list(TARGETS)].to_numpy())

    trained: dict[str, np.ndarray] = {}
    for target in args.targets:
        if target not in TARGETS:
            raise SystemExit(f"unknown target: {target}")
        print(f"\n=== training {target} ===")
        result = train_target(
            target=target,
            y_train=ds.y_train[target],
            X_train=ds.X_train,
            X_test=ds.X_test,
            cfg=cfg,
            model_names=model_names,
            verbose=True,
        )
        trained[target] = result.test
        np.save(args.artifacts / f"oof_{target_slug(target)}.npy", result.oof)
        np.save(args.artifacts / f"test_{target_slug(target)}.npy", result.test)
        report["per_target"][target] = {
            "oof_rmse": result.rmse,
            "per_model_rmse": result.per_model_rmse,
            "weights": result.weights,
        }
        print(f"  OOF RMSE {target}: {result.rmse:.4f} (blended)")
        for m, r in result.per_model_rmse.items():
            print(f"    {m:8s} rmse={r:.4f}  weight={result.weights[m]:.3f}")

    # Compute composite CV score (averaged RMSE over 3 targets) using the SI
    # strategy that wins on OOF — see predict.py for the same logic.
    if all(t in args.targets for t in TARGETS):
        oof_ic50 = np.load(args.artifacts / f"oof_{target_slug('IC50, mM')}.npy")
        oof_cc50 = np.load(args.artifacts / f"oof_{target_slug('CC50, mM')}.npy")
        oof_si_model = np.load(args.artifacts / f"oof_{target_slug('SI')}.npy")
        y_si = ds.y_train["SI"].to_numpy()

        oof_si_ratio = oof_cc50 / np.clip(oof_ic50, 1e-6, None)
        best_alpha = _search_alpha(y_si, oof_si_model, oof_si_ratio)
        oof_si_blend = best_alpha * oof_si_model + (1 - best_alpha) * oof_si_ratio
        rmse_si_ratio = _rmse(y_si, oof_si_ratio)
        rmse_si_model = _rmse(y_si, oof_si_model)
        rmse_si_blend = _rmse(y_si, oof_si_blend)

        report["si_strategies"] = {
            "best_alpha": best_alpha,
            "rmse_ratio": rmse_si_ratio,
            "rmse_model": rmse_si_model,
            "rmse_blend": rmse_si_blend,
        }
        rmse_ic = float(report["per_target"]["IC50, mM"]["oof_rmse"])
        rmse_cc = float(report["per_target"]["CC50, mM"]["oof_rmse"])
        report["composite_cv"] = {
            "with_model_si": (rmse_ic + rmse_cc + rmse_si_model) / 3,
            "with_ratio_si": (rmse_ic + rmse_cc + rmse_si_ratio) / 3,
            "with_blend_si": (rmse_ic + rmse_cc + rmse_si_blend) / 3,
        }

    (args.artifacts / "cv_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nartifacts -> {args.artifacts}")
    print(json.dumps(report, indent=2, default=str))


def _slug(target: str) -> str:
    return target.replace(", ", "_").replace(" ", "_")


def _search_alpha(y_true: np.ndarray, p_model: np.ndarray, p_ratio: np.ndarray) -> float:
    best, best_rmse = 0.0, float("inf")
    for alpha in np.linspace(0.0, 1.0, 51):
        rmse = _rmse(y_true, alpha * p_model + (1 - alpha) * p_ratio)
        if rmse < best_rmse:
            best, best_rmse = float(alpha), float(rmse)
    return best


if __name__ == "__main__":
    main()
