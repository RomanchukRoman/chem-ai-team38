"""Reproducible data loader for ChemAI hackathon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"

TARGETS = ("IC50, mM", "CC50, mM", "SI")
INDEX_COL = "index"


def target_slug(target: str) -> str:
    """Stable filename-safe slug for a target name."""
    return target.replace(", ", "_").replace(" ", "_")


@dataclass(frozen=True)
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    test_index: pd.Series
    feature_cols: list[str]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", decimal=".")


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read(data_dir / "train.csv"), _read(data_dir / "test.csv")


def deduplicate_by_features(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Aggregate exact-feature duplicates: take median of targets.

    Discovered in EDA: 181 fully-duplicated rows differ only in their
    measured targets — same molecule, different lab measurements.
    Median is the robust aggregator.
    """
    return train.groupby(feature_cols, as_index=False, sort=False)[list(TARGETS)].median()


def load_dataset(data_dir: Path = DATA_DIR, *, dedupe: bool = True) -> Dataset:
    train, test = load_raw(data_dir)
    feature_cols = [c for c in train.columns if c not in (*TARGETS, INDEX_COL)]

    if dedupe:
        train = deduplicate_by_features(train, feature_cols)

    X_train = train[feature_cols].copy()
    y_train = train[list(TARGETS)].copy()
    X_test = test[feature_cols].copy()
    test_index = test[INDEX_COL].copy() if INDEX_COL in test.columns else pd.Series(range(len(test)))

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        test_index=test_index,
        feature_cols=feature_cols,
    )
