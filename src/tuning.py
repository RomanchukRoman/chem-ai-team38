"""Optuna hyperparameter search per (model, target).

Each study uses a small KFold CV (3 splits) inside the objective to keep
trials cheap while still penalising overfitting. We tune in log1p space —
identical to production training — so the search result is directly usable.

Run:
    uv run python -m src.tuning --model catboost --target "CC50, mM" --timeout 1500
    uv run python -m src.tuning --model lightgbm --target "IC50, mM" --n-trials 40

Best params are written to artifacts/best_params/{model}_{target_slug}.json
and picked up automatically by src.models.make_*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from src.data import TARGETS, load_dataset
from src.features import build_preprocessor
from src.models import _fit_with_early_stop, _rmse, _slug, make_catboost, make_lightgbm, make_xgboost
from src.seeds import SEED, set_seed

BEST_PARAMS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "best_params"


def _catboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 1500, 4500, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
    }


def _lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1500, 4500, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample_freq": 1,
    }


def _xgboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1500, 4500, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "catboost": _catboost_space,
    "lightgbm": _lightgbm_space,
    "xgboost": _xgboost_space,
}

FACTORIES = {
    "catboost": make_catboost,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
}


def _cv_rmse(
    model_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> float:
    y_values = y.to_numpy(dtype=float)
    y_log = np.log1p(y_values)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y_values)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), 1):
        pp = build_preprocessor(random_state=seed)
        X_tr = pp.fit_transform(X.iloc[tr_idx])
        X_va = pp.transform(X.iloc[va_idx])

        model = FACTORIES[model_name](seed=seed + fold, params=params)
        _fit_with_early_stop(model, X_tr, y_log[tr_idx], X_va, y_log[va_idx], name=model_name)
        oof[va_idx] = np.clip(np.expm1(model.predict(X_va)), 1e-6, None)

    return _rmse(y_values, oof)


def _objective(model_name: str, X: pd.DataFrame, y: pd.Series, *, n_splits: int, seed: int):
    def f(trial: optuna.Trial) -> float:
        params = SPACES[model_name](trial)
        return _cv_rmse(model_name, params, X, y, n_splits=n_splits, seed=seed)

    return f


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna HPO for ChemAI booster.")
    p.add_argument("--model", choices=list(SPACES), required=True)
    p.add_argument("--target", choices=list(TARGETS), required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--timeout", type=int, default=None, help="Wall-clock seconds")
    p.add_argument("--n-splits", type=int, default=3, help="CV folds inside trial")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    y = ds.y_train[args.target]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=f"{args.model}_{_slug(args.target)}",
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    study.optimize(
        _objective(args.model, ds.X_train, y, n_splits=args.n_splits, seed=args.seed),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    out = BEST_PARAMS_DIR / f"{args.model}_{_slug(args.target)}.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "target": args.target,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            },
            indent=2,
        )
    )
    print(f"\nbest RMSE = {study.best_value:.4f}")
    print(f"best params -> {out}")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
