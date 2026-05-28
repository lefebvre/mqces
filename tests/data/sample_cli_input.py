#!/usr/bin/env python3
"""Emit a small JSON file mqces_cli can read, drawn from the fixture data.

Usage:
    .venv/bin/python tests/data/sample_cli_input.py > tests/data/sample_input.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_fixtures import (
    FEATURE_WEIGHTS,
    N_CLASSES,
    N_FEATURES,
    N_SPECIMENS,
    SEED,
    make_class,
)

import numpy as np

rng_classes = np.random.default_rng(SEED)
class_data  = [make_class(k, rng_classes) for k in range(N_CLASSES)]

# Pick test sample drawn from class 10 (mid-range, like the paper).
rng_test = np.random.default_rng(SEED ^ 0xDEADBEEF)
test = [make_class(k, rng_test) for k in range(N_CLASSES)][10]

doc = {
    "weights": FEATURE_WEIGHTS.tolist(),
    "test":    test.tolist(),
    "classes": [
        {"name": f"t={k}", "specimens": class_data[k].tolist()}
        for k in range(N_CLASSES)
    ],
    "options": {
        "epsilon":        0.01,
        "mc_samples":     5,
        "seed":           42,
        "nota_threshold": 0.05,
        "n_threads":      1,
        "variant":        "classic",
    },
}

json.dump(doc, sys.stdout)
sys.stdout.write("\n")
