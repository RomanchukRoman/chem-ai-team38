"""Model factory + ensemble trainer.

Two training modes:
    log1p     — train on log1p(y), invert with expm1 at inference. Optimizes
                relative error; the historical default in our pipeline.
    huber_raw — train on raw y with Huber loss. Optimizes absolute error
                directly (which is what the Kaggle metric, raw RMSE, scores).
                Robust to outliers because Huber is MAE-like above |residual|>δ.

Why bother with huber_raw at all:
    Diagnostics showed our log1p models were only ~15 RMSE better than a
    constant-mean predictor on IC50 (344 vs 359). The reason is that log1p
    minimisation under-weights large-target errors — and those few large
    targets dominate the raw-RMSE score. Switching to a Huber-on-raw loss
    aligns the training objective with the metric.

Each factory accepts an optional `params` dict (best_params from Optuna or
manual overrides). If a Huber-mode factory is built and `huber_delta` is
None, the call site is expected to set it from per-target stats before
training begins.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.data import target_slug
from src.features import build_preprocessor

BEST_PARAMS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "best_params"

LossMode = Literal["log1p", "huber_raw"]


class _Regressor(Protocol):
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...


ModelFactory = Callable[..., _Regressor]


def load_best_params(model_name: str, target: str) -> dict[str, Any]:
    """Return tuned hyperparameters for (model, target) if Optuna has produced any."""
    path = BEST_PARAMS_DIR / f"{model_name}_{target_slug(target)}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("best_params", {})


def make_catboost(
    seed: int,
    params: dict[str, Any] | None = None,
    *,
    loss_mode: LossMode = "log1p",
    huber_delta: float | None = None,
) -> CatBoostRegressor:
    base: dict[str, Any] = {
        "iterations": 4000,
        "learning_rate": 0.02,
        "depth": 5,
        "l2_leaf_reg": 5,
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if loss_mode == "huber_raw":
        if huber_delta is None:
            raise ValueError("huber_delta required for loss_mode='huber_raw'")
        base["loss_function"] = f"Huber:delta={huber_delta}"
    else:
        base["loss_function"] = "RMSE"
    if params:
        base.update(params)
    return CatBoostRegressor(**base)


def make_lightgbm(
    seed: int,
    params: dict[str, Any] | None = None,
    *,
    loss_mode: LossMode = "log1p",
    huber_delta: float | None = None,
) -> LGBMRegressor:
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
        "random_state": seed,
        "verbose": -1,
        "metric": "rmse",
    }
    if loss_mode == "huber_raw":
        if huber_delta is None:
            raise ValueError("huber_delta required for loss_mode='huber_raw'")
        base["objective"] = "huber"
        # LightGBM uses `alpha` as the Huber transition threshold.
        base["alpha"] = float(huber_delta)
    else:
        base["objective"] = "regression"
    if params:
        base.update(params)
    return LGBMRegressor(**base)


def make_xgboost(
    seed: int,
    params: dict[str, Any] | None = None,
    *,
    loss_mode: LossMode = "log1p",
    huber_delta: float | None = None,
) -> XGBRegressor:
    base: dict[str, Any] = {
        "n_estimators": 4000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 3,
        "reg_lambda": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": seed,
        "verbosity": 0,
    }
    if loss_mode == "huber_raw":
        if huber_delta is None:
            raise ValueError("huber_delta required for loss_mode='huber_raw'")
        base["objective"] = "reg:pseudohubererror"
        base["huber_slope"] = float(huber_delta)
    else:
        base["objective"] = "reg:squarederror"
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
    loss_mode: LossMode
    huber_delta: float | None
    train_p99: float
    clip_lo: float
    clip_hi: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _inverse_rmse_weights(per_model_rmse: dict[str, float]) -> dict[str, float]:
    inv = {m: 1.0 / max(r, 1e-9) for m, r in per_model_rmse.items()}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


def compute_clip_bounds(y_train: pd.Series, *, q_hi: float = 0.99) -> tuple[float, float]:
    """Lower/upper clip bounds derived from training targets.

    Lower bound is always 0 — IC50/CC50/SI are physical concentrations and
    cannot be negative. Upper bound is a high quantile (p99 by default) of
    the training distribution: test molecules are drawn from the same lab
    pipeline, so the test distribution should not exceed it materially.
    Capping there absorbs catastrophic extrapolation without losing real
    high values.
    """
    upper = float(np.quantile(y_train.dropna().to_numpy(dtype=float), q_hi))
    return 0.0, upper


def _delta_for_huber(y_train: pd.Series) -> float:
    """Default Huber transition δ = IQR/2 of the training target.

    Robust statistic, immune to the very outliers Huber is meant to handle.
    Roughly: residuals below δ are treated quadratically (MSE), residuals
    above δ are treated linearly (MAE).
    """
    y = y_train.dropna().to_numpy(dtype=float)
    q25, q75 = np.quantile(y, [0.25, 0.75])
    return float(max(1.0, (q75 - q25) / 2.0))


def train_target(
    target: str,
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    cfg: FoldConfig,
    loss_mode: LossMode = "log1p",
    huber_delta: float | None = None,
    clip_q_hi: float = 0.99,
    model_names: tuple[str, ...] = ("catboost", "lightgbm", "xgboost"),
    use_best_params: bool = True,
    verbose: bool = True,
) -> TargetResult:
    """Train an ensemble for a single target via KFold × seeds × models.

    Inference space is always raw target — `log1p` mode applies expm1 inside
    each fold; `huber_raw` mode trains directly on raw and clips at 0 from
    below. Both modes then clip the test predictions at the global upper
    bound from `compute_clip_bounds`.
    """
    y_values = y_train.to_numpy(dtype=float)
    n_train = len(X_train)
    n_test = len(X_test)

    clip_lo, clip_hi = compute_clip_bounds(y_train, q_hi=clip_q_hi)
    train_p99 = clip_hi

    if loss_mode == "huber_raw" and huber_delta is None:
        huber_delta = _delta_for_huber(y_train)
        if verbose:
            print(f"  [{target}] auto huber_delta = {huber_delta:.2f}")

    # Target representation used for fitting; `_postprocess_pred` brings
    # predictions back to raw space for both OOF accumulation and test.
    if loss_mode == "log1p":
        y_fit = np.log1p(y_values)
    elif loss_mode == "huber_raw":
        y_fit = y_values
    else:
        raise ValueError(f"unknown loss_mode: {loss_mode}")

    def _postprocess(pred: np.ndarray) -> np.ndarray:
        if loss_mode == "log1p":
            pred = np.expm1(pred)
        # Clip at [lo, hi]. lo=0 (physical floor), hi=train p99 (anti-extrapolation cap).
        return np.clip(pred, clip_lo, clip_hi)

    oof_per_model = {name: np.zeros(n_train, dtype=float) for name in model_names}
    test_per_model = {name: np.zeros(n_test, dtype=float) for name in model_names}

    overrides: dict[str, dict[str, Any]] = {
        name: (load_best_params(name, target) if use_best_params else {}) for name in model_names
    }
    if verbose:
        for name in model_names:
            tag = "tuned" if overrides[name] else "default"
            print(f"  [{target}] {name:8s} -> {tag}  loss={loss_mode}")

    for seed in cfg.seeds:
        cv = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train), 1):
            preprocessor = build_preprocessor(random_state=cfg.preprocessor_seed)
            X_tr_raw = X_train.iloc[tr_idx]
            X_va_raw = X_train.iloc[va_idx]
            y_tr = y_fit[tr_idx]
            y_va = y_fit[va_idx]

            X_tr = preprocessor.fit_transform(X_tr_raw)
            X_va = preprocessor.transform(X_va_raw)
            X_te = preprocessor.transform(X_test)

            for name in model_names:
                model = MODELS[name](
                    seed=seed + fold,
                    params=overrides[name],
                    loss_mode=loss_mode,
                    huber_delta=huber_delta,
                )
                _fit_with_early_stop(model, X_tr, y_tr, X_va, y_va, name=name)
                va_pred = _postprocess(model.predict(X_va))
                te_pred = _postprocess(model.predict(X_te))
                oof_per_model[name][va_idx] += va_pred / len(cfg.seeds)
                test_per_model[name] += te_pred / (cfg.n_splits * len(cfg.seeds))

            if verbose:
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
        loss_mode=loss_mode,
        huber_delta=huber_delta,
        train_p99=train_p99,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )
