"""Assemble a submission CSV from cached training artifacts.

Strategy semantics (`--si`):
    ratio   — SI = clip(CC50_pred / IC50_pred, lo, hi). Only requires the
              IC50/CC50 test predictions. This is the default.
    model   — Read SI test predictions saved by a previous --train-si run.
              Fails loudly if there's no SI artifact.
    blend   — α-mix of model and ratio with α from cv_report.json. Same
              requirement as `model`.

Run:
    uv run python -m src.predict --si ratio  --out submissions/submission_v3_ratio.csv
    uv run python -m src.predict --si blend  --out submissions/submission_v3_blend.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TARGETS, target_slug

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build submission CSV from artifacts.")
    p.add_argument(
        "--si",
        choices=["ratio", "model", "blend"],
        default="ratio",
        help="How to derive the SI prediction.",
    )
    p.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submissions" / "submission.csv",
    )
    return p.parse_args()


def _load_clip(artifacts: Path) -> dict[str, tuple[float, float]]:
    path = artifacts / "clip_bounds.json"
    if not path.exists():
        # Backwards-compat with old artifacts: no clipping
        return {t: (0.0, float("inf")) for t in TARGETS}
    data = json.loads(path.read_text())
    return {t: (float(lo), float(hi)) for t, (lo, hi) in data.items()}


def main() -> None:
    args = _parse_args()
    bounds = _load_clip(args.artifacts)
    test_index = np.load(args.artifacts / "test_index.npy")
    test_ic50 = np.load(args.artifacts / f"test_{target_slug('IC50, mM')}.npy")
    test_cc50 = np.load(args.artifacts / f"test_{target_slug('CC50, mM')}.npy")

    test_ic50 = np.clip(test_ic50, *bounds["IC50, mM"])
    test_cc50 = np.clip(test_cc50, *bounds["CC50, mM"])

    si_path = args.artifacts / f"test_{target_slug('SI')}.npy"
    si_lo, si_hi = bounds["SI"]
    si_ratio = np.clip(test_cc50 / np.clip(test_ic50, 1e-6, None), si_lo, si_hi)

    if args.si == "ratio":
        si = si_ratio
    elif args.si == "model":
        if not si_path.exists():
            raise SystemExit("test_SI.npy not found — train with --train-si first")
        si = np.clip(np.load(si_path), si_lo, si_hi)
    else:  # blend
        if not si_path.exists():
            raise SystemExit("test_SI.npy not found — train with --train-si first")
        report = json.loads((args.artifacts / "cv_report.json").read_text())
        alpha = float(report.get("si_strategies", {}).get("best_alpha", 0.5))
        si_model = np.clip(np.load(si_path), si_lo, si_hi)
        si = np.clip(alpha * si_model + (1 - alpha) * si_ratio, si_lo, si_hi)
        print(f"using α={alpha:.3f} from cv_report.json")

    sub = pd.DataFrame(
        {
            "index": test_index,
            "IC50": test_ic50,
            "CC50": test_cc50,
            "SI": si,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"wrote {len(sub)} rows to {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
