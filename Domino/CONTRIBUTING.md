# Contributing to Domino

Contributions are welcome when they preserve statistical correctness,
reproducibility, and bounded-memory execution.

## Before opening a change

- Open an issue for a new statistical model or a change to output semantics.
- Keep private genotypes, phenotypes, credentials, and absolute local paths
  out of issues, tests, commits, and logs.
- Use synthetic or openly redistributable test data.
- Keep changes focused and document any numerical or behavioral difference.

## Development setup

```bash
python -m pip install -e ".[test,plot]"
python -m pytest -q
python -m ruff check .
python release_check.py
```

Pull requests that alter model fitting should include a direct numerical
reference test, a null-calibration test where appropriate, and a description
of the expected memory and runtime effect. Changes to output columns or CLI
arguments must update the README and relevant file in `docs/`.

## Reporting results

Report the Domino version, exact command, Python version, decomposition,
variance estimator, precision, memory budget, genotype-cell threshold, and
whether output was streamed. Include the execution JSON when filing a
performance or approximation issue, after checking it for sensitive paths.
