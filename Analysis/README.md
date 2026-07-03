# Analysis guide

## Confirmatory R workflow

[`analysis_R.qmd`](analysis_R.qmd) implements the published analysis: nonlinear beta regression of the hyperboloid discounting functions, posterior summaries, empirical AUC, and mixed-effects beta regression for amount effects. It uses the article's positive Normal(1, 10) priors for the discounting parameters and the reported four-chain sampling schedule.

The article reports R 4.2.1. The repository's `renv.lock` records a later R 4.4.1 replication environment; retain the generated session information when reporting reproduced estimates.

Run from the repository root:

```bash
quarto render Analysis/analysis_R.qmd
```

## Independent Python workflow

[`analysis_python.py`](analysis_python.py) implements the same data transformations, equations, and priors with PyMC. The Python amount-effect model is Bayesian and therefore does not reproduce the article's frequentist p-values. Use the R workflow for confirmatory numerical replication.

```bash
python Analysis/analysis_python.py --data-dir data
```

The notebook [`analysis_Py.ipynb`](analysis_Py.ipynb) provides an interactive interface to the module.

## Outputs

Both workflows write generated artifacts below `outputs/`; the directory is excluded from Git.

```text
outputs/
├── figures/
├── models/
└── tables/
```

Do not interpret a model until all chains have converged, split rank-normalized R-hat values are acceptably close to 1.00, effective sample sizes are adequate, and there are no unresolved divergent transitions.
