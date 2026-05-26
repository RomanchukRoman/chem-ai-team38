"""Model factory + ensemble trainer.

Each factory builds a regressor with sensible defaults, then merges any
overrides from `params=...`. If `params` is None and a JSON file at
`artifacts/best_params/{model}_{target_slug}.json` exists, those tuned
values are loaded automatically — so once Optuna has run, the training
script picks up the better hyperparameters with zero further wiring.

Why log1p on the target?
    All three taggets are heavily right-skewed (IC50 skew=3.79, CC50=2.06,
    SI=15.63). Training in log space makes the loss roughly homoscedastic;
    `np.clip(..., 1e-6, None)` after `expm1` keeps the SI ratio safe.

Why an inverse-RMSE weighted blend across boosters?
    Simple averaging across CatBoost/LightGBM/XGBoost actually hurt CC50
    in our baseline (mean=476.5 vs best-single=472.8) because the worst
    booster diluted the best one. Weighting by OOF accuracy moves us
    toward whichever booster wins that specific target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.data import target_slug
from src.features import build_preprocessor

BEST_PARAMS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "best_params"


class _Regressor(Protocol):
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...


ModelFactory = Callable[..., _Regressor]


# Backwards-compatible alias used by src.train / src.predict
def _slug(target: str) -> str:
    return target_slug(target)


def load_best_params(model_name: str, target: str) -> dict[str, Any]:
    """Return tuned hyperparameters for (model, target) if Optuna has produced any."""
    path = BEST_PARAMS_DIR / f"{model_name}_{target_slug(target)}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("best_params", {})


def make_catboost(seed: int, params: dict[str, Any] | None = None) -> CatBoostRegressor:
    base: dict[str, Any] = {
        "iterations": 4000,
        "learning_rate": 0.02,
        "depth": 5,
        "l2_leaf_reg": 5,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if params:
        base.update(params)
    return CatBoostRegressor(**base)


def make_lightgbm(seed: int, params: dict[str, Any] | None = None) -> LGBMRegressor:
    base: dict[str, Any] = {
        "n_estimators": 4000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 10,
        "reg_lambda": 5.0,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "objective": "regression",
        "random_state": seed,
        "verbose": -1,
    }
    if params:
        base.update(params)
    return LGBMRegressor(**base)


def make_xgboost(seed: int, params: dict[str, Any] | None = None) -> XGBRegressor:
    base: dict[str, Any] = {
        "n_estimators": 4000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 3,
        "reg_lambda": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": seed,
        "verbosity": 0,
    }
    if params:
        base.update(params)
    return XGBRegressor(**base)


MODELS: dict[str, ModelFactory] = {
    "catboost": make_catboost,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
}


def _fit_with_early_stop(model, X_tr, y_tr, X_val, y_val, *, name: str) -> None:
    """Each library wants slightly different fit arguments — unify here."""
    if name == "catboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=300,
            use_best_model=True,
        )
    elif name == "lightgbm":
        from lightgbm import early_stopping

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=300, verbose=False)],
        )
    elif name == "xgboost":
        model.set_params(early_stopping_rounds=300)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        raise ValueError(f"unknown model: {name}")


@dataclass
class FoldConfig:
    n_splits: int = 5
    seeds: tuple[int, ...] = (42, 77, 123)
    preprocessor_seed: int = 42


@dataclass
class TargetResult:
    target: str
    oof: np.ndarray
    test: np.ndarray
    rmse: float
    per_model_rmse: dict[str, float]
    weights: dict[str, float]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _inverse_rmse_weights(per_model_rmse: dict[str, float]) -> dict[str, float]:
    inv = {m: 1.0 / max(r, 1e-9) for m, r in per_model_rmse.items()}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


def train_target(
    target: str,
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    cfg: FoldConfig,
    model_names: tuple[str, ...] = ("catboost", "lightgbm", "xgboost"),
    use_best_params: bool = True,
    verbose: bool = True,
) -> TargetResult:
    """Train an ensemble for a single target via KFold × seeds × models."""
    y_values = y_train.to_numpy(dtype=float)
    y_log = np.log1p(y_values)
    n_train = len(X_train)
    n_test = len(X_test)

    oof_per_model = {name: np.zeros(n_train, dtype=float) for name in model_names}
    test_per_model = {name: np.zeros(n_test, dtype=float) for name in model_names}

    overrides: dict[str, dict[str, Any]] = {
        name: (load_best_params(name, target) if use_best_params else {}) for name in model_names
    }
    if verbose:
        for name in model_names:
            tag = "tuned" if overrides[name] else "default"
            print(f"  [{target}] {name:8s} -> {tag}")

    for seed in cfg.seeds:
        cv = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train), 1):
            preprocessor = build_preprocessor(random_state=cfg.preprocessor_seed)
            X_tr_raw = X_train.iloc[tr_idx]
            X_va_raw = X_train.iloc[va_idx]
            y_tr = y_log[tr_idx]
            y_va = y_log[va_idx]

            X_tr = preprocessor.fit_transform(X_tr_raw)
            X_va = preprocessor.transform(X_va_raw)
            X_te = preprocessor.transform(X_test)

            for name in model_names:
                model = MODELS[name](seed=seed + fold, params=overrides[name])
                _fit_with_early_stop(model, X_tr, y_tr, X_va, y_va, name=name)
                va_pred = np.clip(np.expm1(model.predict(X_va)), 1e-6, None)
                te_pred = np.clip(np.expm1(model.predict(X_te)), 1e-6, None)
                oof_per_model[name][va_idx] += va_pred / len(cfg.seeds)
                test_per_model[name] += te_pred / (cfg.n_splits * len(cfg.seeds))

            if verbose:
                # noisy fold-level snapshot — the full OOF below is the truth
                rough = np.mean(
                    [oof_per_model[m][va_idx] * len(cfg.seeds) for m in model_names], axis=0
                )
                print(f"  [{target}] seed={seed} fold={fold}  fold-RMSE≈{_rmse(y_values[va_idx], rough):.3f}")

    per_model_rmse = {name: _rmse(y_values, oof_per_model[name]) for name in model_names}
    weights = _inverse_rmse_weights(per_model_rmse)
    oof_blend = sum(weights[m] * oof_per_model[m] for m in model_names)
    test_blend = sum(weights[m] * test_per_model[m] for m in model_names)

    return TargetResult(
        target=target,
        oof=oof_blend,
        test=test_blend,
        rmse=_rmse(y_values, oof_blend),
        per_model_rmse=per_model_rmse,
        weights=weights,
    )
