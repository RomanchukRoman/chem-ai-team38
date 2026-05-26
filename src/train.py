"""End-to-end training entrypoint.

Trains the IC50 and CC50 ensembles. SI is NOT modelled by default — our
diagnostics showed our SI model was barely beating a constant baseline
(105 vs 107 RMSE), so it doesn't earn its compute. SI is derived at
predict time as `clip(CC50_pred / IC50_pred, 0, p99_SI)`.

Run:
    uv run python -m src.train                       # huber_raw on IC50/CC50, no SI
    uv run python -m src.train --loss log1p          # legacy log1p mode
    uv run python -m src.train --train-si            # also train SI separately
    uv run python -m src.train --quick               # 1 seed × 3 splits × catboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import TARGETS, load_dataset, target_slug
from src.models import (
    FoldConfig,
    LossMode,
    _rmse,
    compute_clip_bounds,
    train_target,
)
from src.seeds import SEED, set_seed

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ChemAI ensemble.")
    p.add_argument("--n-seeds", type=int, default=3, help="Number of seeds per target.")
    p.add_argument("--n-splits", type=int, default=5, help="KFold splits.")
    p.add_argument(
        "--loss",
        choices=["huber_raw", "log1p"],
        default="huber_raw",
        help="Loss mode for IC50/CC50.",
    )
    p.add_argument(
        "--train-si",
        action="store_true",
        help="Also train a separate SI model (default off).",
    )
    p.add_argument(
        "--si-loss",
        choices=["huber_raw", "log1p"],
        default="log1p",
        help="Loss mode if SI is trained (default log1p — SI has skew=11).",
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

    targets_to_train: list[tuple[str, LossMode]] = [
        ("IC50, mM", args.loss),
        ("CC50, mM", args.loss),
    ]
    if args.train_si:
        targets_to_train.append(("SI", args.si_loss))

    report: dict[str, object] = {
        "config": {
            "seed": args.seed,
            "seeds": list(seeds),
            "n_splits": n_splits,
            "model_names": list(model_names),
            "targets_to_train": [(t, m) for t, m in targets_to_train],
            "quick": args.quick,
        },
        "per_target": {},
    }

    np.save(args.artifacts / "test_index.npy", ds.test_index.to_numpy())
    np.save(args.artifacts / "y_train.npy", ds.y_train[list(TARGETS)].to_numpy())

    # Save clip bounds for all 3 targets up front — predict.py uses the SI
    # bound even though we don't train an SI model.
    clip_bounds: dict[str, tuple[float, float]] = {}
    for tgt in TARGETS:
        lo, hi = compute_clip_bounds(ds.y_train[tgt])
        clip_bounds[tgt] = (lo, hi)
    (args.artifacts / "clip_bounds.json").write_text(
        json.dumps({k: list(v) for k, v in clip_bounds.items()}, indent=2)
    )

    trained_targets: set[str] = set()

    for target, loss_mode in targets_to_train:
        if target not in TARGETS:
            raise SystemExit(f"unknown target: {target}")
        print(f"\n=== training {target}  (loss={loss_mode}) ===")
        result = train_target(
            target=target,
            y_train=ds.y_train[target],
            X_train=ds.X_train,
            X_test=ds.X_test,
            cfg=cfg,
            loss_mode=loss_mode,
            model_names=model_names,
            verbose=True,
        )
        np.save(args.artifacts / f"oof_{target_slug(target)}.npy", result.oof)
        np.save(args.artifacts / f"test_{target_slug(target)}.npy", result.test)
        trained_targets.add(target)
        report["per_target"][target] = {
            "loss_mode": result.loss_mode,
            "huber_delta": result.huber_delta,
            "clip_lo": result.clip_lo,
            "clip_hi": result.clip_hi,
            "oof_rmse": result.rmse,
            "per_model_rmse": result.per_model_rmse,
            "weights": result.weights,
        }
        print(f"  OOF RMSE {target}: {result.rmse:.4f} (blended, clip≤{result.clip_hi:.1f})")
        for m, r in result.per_model_rmse.items():
            print(f"    {m:8s} rmse={r:.4f}  weight={result.weights[m]:.3f}")

    # Composite CV — assemble whatever components we have.
    if {"IC50, mM", "CC50, mM"}.issubset(trained_targets):
        oof_ic50 = np.load(args.artifacts / f"oof_{target_slug('IC50, mM')}.npy")
        oof_cc50 = np.load(args.artifacts / f"oof_{target_slug('CC50, mM')}.npy")
        y_si = ds.y_train["SI"].to_numpy()
        si_lo, si_hi = clip_bounds["SI"]

        oof_si_ratio = np.clip(oof_cc50 / np.clip(oof_ic50, 1e-6, None), si_lo, si_hi)
        rmse_si_ratio = _rmse(y_si, oof_si_ratio)

        rmse_ic = float(report["per_target"]["IC50, mM"]["oof_rmse"])
        rmse_cc = float(report["per_target"]["CC50, mM"]["oof_rmse"])
        composite: dict[str, float] = {
            "with_ratio_si": (rmse_ic + rmse_cc + rmse_si_ratio) / 3,
        }
        report["si_strategies"] = {"rmse_ratio": rmse_si_ratio}

        if "SI" in trained_targets:
            oof_si_model = np.load(args.artifacts / f"oof_{target_slug('SI')}.npy")
            rmse_si_model = _rmse(y_si, oof_si_model)
            best_alpha = _search_alpha(y_si, oof_si_model, oof_si_ratio)
            oof_si_blend = best_alpha * oof_si_model + (1 - best_alpha) * oof_si_ratio
            oof_si_blend = np.clip(oof_si_blend, si_lo, si_hi)
            rmse_si_blend = _rmse(y_si, oof_si_blend)
            report["si_strategies"].update(
                {
                    "rmse_model": rmse_si_model,
                    "rmse_blend": rmse_si_blend,
                    "best_alpha": best_alpha,
                }
            )
            composite["with_model_si"] = (rmse_ic + rmse_cc + rmse_si_model) / 3
            composite["with_blend_si"] = (rmse_ic + rmse_cc + rmse_si_blend) / 3
        report["composite_cv"] = composite

    (args.artifacts / "cv_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nartifacts -> {args.artifacts}")
    print(json.dumps(report, indent=2, default=str))


def _search_alpha(y_true: np.ndarray, p_model: np.ndarray, p_ratio: np.ndarray) -> float:
    best, best_rmse = 0.0, float("inf")
    for alpha in np.linspace(0.0, 1.0, 51):
        rmse = _rmse(y_true, alpha * p_model + (1 - alpha) * p_ratio)
        if rmse < best_rmse:
            best, best_rmse = float(alpha), float(rmse)
    return best


if __name__ == "__main__":
    main()
