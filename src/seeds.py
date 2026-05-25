"""Single source of truth for randomness — call set_seed(SEED) at every entrypoint."""

from __future__ import annotations

import os
import random

import numpy as np

SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
