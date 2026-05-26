#!/usr/bin/env bash
# Run full Optuna HPO for all (model, target) combos sequentially.
# Each study writes artifacts/best_params/{model}_{slug}.json. Re-running is
# safe — JSON files are overwritten with the latest best.

set -euo pipefail

cd "$(dirname "$0")/.."

TRIALS_CC50=40   # main lever for the score
TRIALS_IC50=30
TRIALS_SI=20     # tiny weight on composite — quick budget

for model in catboost lightgbm xgboost; do
    echo "=========== $model | CC50 ==========="
    uv run python -m src.tuning --model "$model" --target "CC50, mM" --n-trials "$TRIALS_CC50"

    echo "=========== $model | IC50 ==========="
    uv run python -m src.tuning --model "$model" --target "IC50, mM" --n-trials "$TRIALS_IC50"

    echo "=========== $model | SI ==========="
    uv run python -m src.tuning --model "$model" --target "SI" --n-trials "$TRIALS_SI"
done

echo "All HPO studies done."
ls -la artifacts/best_params/
