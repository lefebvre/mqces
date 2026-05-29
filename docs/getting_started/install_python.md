# Installing the Python bindings

## From source

```bash
cmake -B build -DMQCES_ENABLE_PYTHON=ON
cmake --build build -j
PYTHONPATH=build/python python -m pytest python/tests
```

The build produces `_mqces_core.so` (nanobind extension) under
`build/python/`, which the pure-Python wrapper in `python/mqces/__init__.py`
imports.

## Smoke test

```python
import numpy as np
import mqces

rng = np.random.default_rng(0)
test    = rng.normal(size=(50, 9))
class_a = rng.normal(loc=0.0, size=(80, 9))
class_b = rng.normal(loc=1.0, size=(80, 9))
weights = np.ones(9)

result = mqces.classify(
    test=test,
    classes=[("a", class_a), ("b", class_b)],
    weights=weights,
    variant="v4",
)
print(result.best_class, result.misclassification_prob)
```

## API surface

The Python package re-exports:

- {py:func}`mqces.classify` — high-level classifier with variant dispatch.
- {py:func}`mqces.similarity_score` — Eq. 3 primitive.
- {py:func}`mqces.spatial_rank` — Eq. 1 primitive.
- {py:func}`mqces.perturb` — Eq. 10 multiplicative-Gaussian sampler.

See the [Python API reference](../reference/python/index.md) for the full
signature of every exported symbol.
