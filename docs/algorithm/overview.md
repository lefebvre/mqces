# Algorithm overview

This page summarizes the Weber & Dayman classification method as
implemented in mqces. For the full derivation and motivation, consult the
paper.

> C. Weber and K. Dayman, *Using Quantile Comparisons to Classify
> Environmental Samples*, Oak Ridge National Laboratory.

## Notation

| Symbol           | Meaning                                                                         |
|------------------|---------------------------------------------------------------------------------|
| $X, Y$           | Samples (clouds) of multivariate measurements.                                  |
| $x_j$            | The $j$-th row of $X$ — a single specimen.                                      |
| $N$              | Number of specimens (rows) in a cloud.                                          |
| $d$              | Number of features (columns) — e.g. isotope ratios.                             |
| $U_j$            | Spatial rank of $x_j$ relative to its cloud (Eq. 1).                            |
| $\tilde{x}_j$    | Inverse-spatial-rank reconstruction of $x_j$ against $Y$ (Eq. 2).               |
| $W$              | Diagonal feature-weight matrix (configurable per call).                         |
| $S_{XY}$         | Similarity score between two clouds (Eq. 3); lower = more similar.              |
| $\varepsilon$    | Relative measurement error fraction (Eq. 10).                                   |

## Decision flow

The high-level pipeline for a single classification call is:

```
   test sample T            known classes [X_1, X_2, ...]
        │                              │
        └──────────┬───────────────────┘
                   ▼
   for each class X_k:
     ┌─────────────────────────────────────────────────┐
     │ 1. Optional perturbation:                       │
     │      T̃ = T  · (1 + ε N(0,1))                   │
     │      X̃_k = X_k · (1 + ε N(0,1))                │
     │    Repeat M = mc_samples times.                 │
     │ 2. For each MC replicate:                       │
     │      s_m = similarity_score(T̃, X̃_k, W)        │
     │ 3. Average:                                     │
     │      S_k = mean(s_1, ..., s_M)                  │
     └─────────────────────────────────────────────────┘
                   │
                   ▼
   sort classes by S_k ascending → best, runner-up
                   │
                   ▼
   compute misclassification_prob from t-test
   (Eq. 6 in classic; paired-diff in v2/v3/v4)
                   │
                   ▼
   none_of_the_above? compare per-MC scores against
   within-class scores split from best.specimens.
                   │
                   ▼
   ClassificationResult
```

## Variant axes

The four shipped variants ([`classic`, `v2`, `v3`, `v4`](../guide/variants.md))
differ along two axes:

- **t-statistic**: `classic` uses the independent-samples form (Eq. 6 of
  the paper); `v2`/`v3`/`v4` use a paired-difference form.
- **Inner solver** (for inverse spatial rank): `classic`/`v2` use
  Weiszfeld; `v3` uses Vardi-Zhang; `v4` adds Type-II Anderson
  acceleration on top of VZ.

Everything else — the `similarity_score` primitive (Eq. 3), the
Monte-Carlo measurement-error sampler (Eq. 10), and the NOTA decision
(page 8 of the paper) — is shared across variants.

## Where to read next

- {doc}`similarity_score` — the inner score primitive, Eq. 3 in detail.
- {doc}`spatial_rank` — forward and inverse spatial rank, Eq. 1 / Eq. 2.
- {doc}`solvers` — Weiszfeld, Vardi-Zhang, and Anderson acceleration
  side-by-side.
