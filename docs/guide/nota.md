# None-of-the-above (NOTA)

A classifier that always picks one of its known classes is useless when the
test sample isn't from any of them. mqces's NOTA decision compares the
test-vs-best similarity score distribution against the *within-class*
distribution of the winning class, and flags the result if the two
populations differ at the configured significance level.

## How it works

After the per-class scores are computed and a best class chosen, mqces:

1. Splits `best.specimens` into halves (no overlap).
2. Perturbs each half via Eq. 10.
3. Computes the similarity score between the two halves; repeats
   `mc_samples` times to get a within-class score distribution.
4. Runs a two-sample t-test comparing the within-class scores to the
   test-vs-best scores already computed in step (1) of the classifier.
5. If the test scores are statistically *larger* than the within-class
   scores at p < `nota_threshold`, returns `none_of_the_above = true`.

## Configuring

```{eval-rst}
.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            mqces::ClassifierOptions opts;
            opts.weights        = weights;
            opts.nota_threshold = 0.05;   // default

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            result = mqces.classify(
                test, classes, weights,
                nota_threshold=0.05,
            )

    .. tab-item:: CLI
        :sync: cli

        .. code-block:: json

            { "options": { "nota_threshold": 0.05 } }
```

## Picking `nota_threshold`

- **0.05** *(default)* — standard significance; rejects classifications
  whose test scores look significantly worse than the class's own
  within-class spread.
- **0.01** — more permissive about classifying borderline samples
  (fewer false NOTA flags, more false accepts).
- **0.10** — more conservative; flags more samples as NOTA. Useful when
  the cost of misclassification dominates.

## When NOTA is bypassed

The function returns `false` (i.e. *accept the classification*) when:

- `best.specimens` has fewer than 4 rows — splitting is meaningless.
- Either distribution has zero spread — no statistical signal.
- The test-vs-best mean is *below* the within-class mean — the test
  sample matches the chosen class better than the class matches itself,
  which is definitely not NOTA.
