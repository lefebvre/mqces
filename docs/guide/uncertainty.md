# Measurement uncertainty (Monte Carlo)

Real samples carry measurement error. mqces propagates this error by drawing
multiple perturbed replicates of each input and averaging the resulting
scores — Equation 10 of Weber & Dayman.

## Equation 10

The perturbation is multiplicative Gaussian, applied element-wise:

$$ X = X^0 \cdot (1 + \varepsilon\, \mathcal{N}(0, 1)). $$

`epsilon` is the relative measurement error. Setting `epsilon = 0` is a
no-op — the classifier sees the inputs unchanged and `mc_samples` is
ignored.

## Configuring

```{eval-rst}
.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            mqces::ClassifierOptions opts;
            opts.weights                = weights;
            opts.uncertainty.epsilon    = 0.01;        // 1% measurement error
            opts.uncertainty.mc_samples = 25;          // 25 replicates per call
            opts.uncertainty.seed       = 0xC0FFEE;    // deterministic; per-thread streams derived

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            result = mqces.classify(
                test, classes, weights,
                epsilon=0.01, mc_samples=25, seed=0xC0FFEE,
            )

    .. tab-item:: CLI
        :sync: cli

        .. code-block:: json

            { "options": { "epsilon": 0.01, "mc_samples": 25, "seed": 49374 } }
```

## Determinism

The `seed` field deterministically selects the RNG stream. Per-thread
streams are derived from it, so the result is reproducible across runs and
across thread counts as long as the same seed is supplied.

## Picking `mc_samples`

The variance of the averaged score scales as $O(1/M)$ where $M$ is
`mc_samples`. Doubling `mc_samples` halves the score variance and doubles
the wall time. The default of 10 is a reasonable starting point; bump it
to 25 – 50 when chasing a tight misclassification probability.

## Direct primitive

If you need the perturbation outside `classify`, call
{func}`mqces.perturb` (Python) or `mqces::perturb` (C++) directly.
