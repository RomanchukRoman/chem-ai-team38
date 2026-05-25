"""Assemble a submission CSV from cached training artifacts.

Run:
    uv run python -m src.predict --si ratio  --out submissions/submission_ratio.csv
    uv run python -m src.predict --si model  --out submissions/submission_model.csv
    uv run python -m src.predict --si blend  --out submissions/submission_blend.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TARGETS

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _slug(target: str) -> str:
    return target.replace(", ", "_").replace(" ", "_")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build submission CSV from artifacts.")
    p.add_argument(
        "--si",
        choices=["ratio", "model", "blend"],
        default="blend",
        help="How to derive the SI prediction.",
    )
    p.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submissions" / "submission.csv",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    test_index = np.load(args.artifacts / "test_index.npy")
    test_ic50 = np.load(args.artifacts / f"test_{_slug('IC50, mM')}.npy")
    test_cc50 = np.load(args.artifacts / f"test_{_slug('CC50, mM')}.npy")

    si_strategy = args.si
    if si_strategy in {"model", "blend"}:
        test_si_model = np.load(args.artifacts / f"test_{_slug('SI')}.npy")
    si_ratio = test_cc50 / np.clip(test_ic50, 1e-6, None)

    if si_strategy == "ratio":
        si = si_ratio
    elif si_strategy == "model":
        si = test_si_model
    else:  # blend — use the alpha discovered during training
        report = json.loads((args.artifacts / "cv_report.json").read_text())
        alpha = float(report.get("si_strategies", {}).get("best_alpha", 0.5))
        si = alpha * test_si_model + (1 - alpha) * si_ratio
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
    # Touch TARGETS so the import isn't flagged as unused in linters that ignore re-exports.
    _ = TARGETS
    main()
