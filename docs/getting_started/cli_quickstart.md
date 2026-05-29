# CLI quickstart

The `mqces` CLI consumes a JSON description of the problem and emits a JSON
classification result. It's the most language-agnostic way to call the
library — useful for shell pipelines, scripted experiments, and language
bindings that don't yet exist.

## Minimal example

`input.json`:

```json
{
  "weights": [1.0, 1.0, 1.0],
  "test": [
    [0.1, 0.2, 0.3],
    [0.2, 0.1, 0.4]
  ],
  "classes": [
    { "name": "alpha", "specimens": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]] },
    { "name": "beta",  "specimens": [[1.0, 1.0, 1.0], [1.1, 0.9, 1.0]] }
  ],
  "options": {
    "variant": "v4",
    "mc_samples": 10,
    "seed": 42
  }
}
```

Run:

```bash
mqces --pretty input.json
```

Output:

```json
{
  "best_class": "alpha",
  "scores": [
    { "class_name": "alpha", "s_xy": 0.0123 },
    { "class_name": "beta",  "s_xy": 1.4567 }
  ],
  "misclassification_prob": 0.0001,
  "none_of_the_above": false,
  "variant": "v4"
}
```

## Useful flags

| Flag                     | Purpose                                                       |
|--------------------------|---------------------------------------------------------------|
| `-o, --output <file>`    | Write JSON result to `<file>` instead of stdout.              |
| `--pretty`               | Pretty-print the JSON output (indent = 2).                    |
| `--variant <name>`       | Override `options.variant` (`classic`, `v2`, `v3`, `v4`).     |
| `--reference-size <N>`   | Override `options.reference_size` (`0` == exact).             |
| `-h, --help`             | Print full usage and exit.                                    |
| `-V, --version`          | Print mqces version and exit.                                 |

## Full JSON schema

See the [CLI reference](../reference/cli.md) for every supported field in
the input and output JSON, including solver tuning knobs and exit codes.
