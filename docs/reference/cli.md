# CLI reference

The `mqces` executable is built when `MQCES_ENABLE_CLI=ON` (the default).
It reads a JSON input file and writes a JSON classification result.

## Usage

```text
mqces [options] <input.json>

Options:
  -o, --output <file>          Write JSON result to <file> (default: stdout).
      --pretty                 Pretty-print the JSON output (indent=2).
      --variant <name>         Override input.options.variant
                               (classic|v2|v3|v4).
      --reference-size <N>     Override input.options.reference_size
                               (0 == exact).
  -h, --help                   Show usage and exit.
  -V, --version                Print mqces version and exit.
```

Exit codes:

| Code | Meaning                                                  |
|------|----------------------------------------------------------|
| 0    | Classification completed; result emitted.                |
| 1    | Bad arguments, unreadable input, or invalid JSON.        |
| 2    | Classifier threw — see stderr for the exception message. |

## Input schema

```json
{
  "weights":  [1.0, 1.0, 1.0],
  "test":     [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]],
  "classes": [
    { "name": "alpha", "specimens": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]] }
  ],
  "options": {
    "epsilon":          0.05,
    "mc_samples":       10,
    "seed":             42,
    "nota_threshold":   0.05,
    "n_threads":        0,
    "variant":          "classic",
    "reference_size":   0,
    "sampling_seed":    180160751,
    "solver_tol":       1e-7,
    "solver_max_iters": 500,
    "vz_vertex_eps":    1e-6,
    "aa_window":        5,
    "aa_reg":           1e-12
  }
}
```

| Field                       | Required | Default     | Notes                                                              |
|-----------------------------|----------|-------------|--------------------------------------------------------------------|
| `weights`                   | yes      | —           | Length must equal `n_features`.                                    |
| `test`                      | yes      | —           | `(n_specimens, n_features)`.                                       |
| `classes[].name`            | yes      | —           | Must be unique across classes.                                     |
| `classes[].specimens`       | yes      | —           | `(n_class, n_features)`; same `n_features` as `test`.              |
| `options.variant`           | no       | `"classic"` | One of `classic`, `v2`, `v3`, `v4`. CLI `--variant` wins.          |
| `options.reference_size`    | no       | `0`         | `0` = exact; `>0` = O(N·R) subsample. CLI `--reference-size` wins. |
| `options.epsilon`           | no       | `0.0`       | Measurement-error fraction (Eq. 10).                               |
| `options.mc_samples`        | no       | `10`        | Monte-Carlo replicate count.                                       |
| `options.seed`              | no       | `12648430`  | RNG seed for Eq. 10.                                               |
| `options.sampling_seed`     | no       | `180160751` | RNG seed for the subsample (when `reference_size > 0`).            |
| `options.nota_threshold`    | no       | `0.05`      | NOTA p-value threshold.                                            |
| `options.n_threads`         | no       | `0`         | `0` = auto. **Mutates process-global OpenMP thread cap.**          |
| `options.solver_tol`        | no       | variant-dep | Inner-solver convergence tolerance.                                |
| `options.solver_max_iters`  | no       | variant-dep | Inner-solver iteration cap.                                        |
| `options.vz_vertex_eps`     | no       | `1e-8`      | Vardi-Zhang subgradient correction radius (v3/v4 only).            |
| `options.aa_window`         | no       | `5`         | Anderson rolling-window size (v4 only).                            |
| `options.aa_reg`            | no       | `1e-12`     | Anderson Tikhonov regularization (v4 only).                        |

## Output schema

```json
{
  "best_class":             "alpha",
  "scores":                 [{ "class_name": "alpha", "s_xy": 0.001 }],
  "misclassification_prob": 0.018,
  "none_of_the_above":      false,
  "variant":                "v4"
}
```

`scores` is sorted ascending by `s_xy`. `scores[0].class_name == best_class`.
