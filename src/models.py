"""Model factory: CatBoost / LightGBM / XGBoost regressors trained on log1p(y).

Why an ensemble of three tree boosters?
    GBDTs differ in how they handle categorical signals, leaf splits and
    regularisation. On a small tabular set (≈628 rows) the variance between
    them dominates the bias of any single one, so a simple unweighted mean of
    their predictions tends to beat each one individually — without
    additional hyperparameter risk from stacking with a meta-learner.

Why log1p?
    All three targets are heavily right-skewed (IC50 skew=3.79, CC50=2.06,
    SI=15.63). Training in log space makes the loss roughly homoscedastic,
    and `np.clip(..., 1e-6, None)` after `expm1` keeps the SI ratio safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.features import build_preprocessor


class _Regressor(Protocol):
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...


ModelFactory = Callable[[int], _Regressor]


def make_catboost(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=4000,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )


def make_lightgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        reg_lambda=5.0,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        objective="regression",
        random_state=seed,
        verbose=-1,
    )


def make_xgboost(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=3,
        reg_lambda=5.0,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
        verbosity=0,
    )


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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_target(
    target: str,
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    cfg: FoldConfig,
    model_names: tuple[str, ...] = ("catboost", "lightgbm", "xgboost"),
    verbose: bool = True,
) -> TargetResult:
    """Train an ensemble for a single target via KFold × seeds × models."""
    y_values = y_train.to_numpy(dtype=float)
    y_log = np.log1p(y_values)
    n_train = len(X_train)
    n_test = len(X_test)

    # OOF predictions per (model, sample) — averaged across seeds.
    oof_per_model = {name: np.zeros(n_train, dtype=float) for name in model_names}
    test_per_model = {name: np.zeros(n_test, dtype=float) for name in model_names}

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
                model = MODELS[name](seed=seed + fold)
                _fit_with_early_stop(model, X_tr, y_tr, X_va, y_va, name=name)
                va_pred = np.clip(np.expm1(model.predict(X_va)), 1e-6, None)
                te_pred = np.clip(np.expm1(model.predict(X_te)), 1e-6, None)
                # Average across seeds (1/n_seeds), test also averages folds.
                oof_per_model[name][va_idx] += va_pred / len(cfg.seeds)
                test_per_model[name] += te_pred / (cfg.n_splits * len(cfg.seeds))

            if verbose:
                fold_rmse = _rmse(
                    y_values[va_idx],
                    np.mean([oof_per_model[m][va_idx] * len(cfg.seeds) for m in model_names], axis=0),
                )
                # NB: this is a noisy fold-level snapshot (only this seed); full OOF below is the truth.
                print(f"  [{target}] seed={seed} fold={fold}  fold-RMSE≈{fold_rmse:.3f}")

    # Final OOF: simple unweighted mean across the 3 boosters.
    oof_blend = np.mean(list(oof_per_model.values()), axis=0)
    test_blend = np.mean(list(test_per_model.values()), axis=0)
    per_model_rmse = {name: _rmse(y_values, oof_per_model[name]) for name in model_names}
    return TargetResult(
        target=target,
        oof=oof_blend,
        test=test_blend,
        rmse=_rmse(y_values, oof_blend),
        per_model_rmse=per_model_rmse,
    )
