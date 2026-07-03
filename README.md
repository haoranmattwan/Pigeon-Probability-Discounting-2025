# Discounting of Probabilistic Food Reinforcement by Pigeons

Reproducible analyses for:

> Oliveira, L., Green, L., Myerson, J., & Wan, H. (2025). Discounting of probabilistic food reinforcement by pigeons. *Journal of the Experimental Analysis of Behavior, 124*, e70042. <https://doi.org/10.1002/jeab.70042>

## Study

This study combined concurrent-chains and adjusting-amount procedures to estimate probability-discounting functions in eight pigeons. The procedure ensured that the probabilities experienced within each session matched the programmed probabilities.

- **Experiment 1:** choice between a smaller, certain reinforcer and a larger, probabilistic reinforcer.
- **Experiment 2:** choice between a smaller, more probable reinforcer and a larger, less probable reinforcer; both probabilities were reduced by a common multiplier.

Across both experiments, relative subjective value decreased as the odds against reinforcement increased. The hyperboloid discounting function described most individual and group-mean data well. The published analyses found no reliable effect of reinforcer amount (16 vs. 32 pellets) on discounting.

## Open materials

Data are maintained on the **[Open Science Framework (OSF)](https://osf.io/scwg3/)**, which is the authoritative source. Download these files from OSF:

- `PigeonPD_Exp1.csv`
- `PigeonPD_Exp2.csv`

Place both files in a local `data/` directory. Data are intentionally excluded from Git to avoid maintaining divergent copies. See [`data/README.md`](data/README.md) for the expected layout and validation rules.

## Analyses

The R analysis is the confirmatory replication of the published statistical workflow. The Python analysis is an independent Bayesian implementation of the same discounting equations and priors; its Bayesian amount-effect model is a companion analysis, not a numerical reproduction of the article's frequentist mixed-effects beta regression.

| Path | Purpose |
| --- | --- |
| [`Analysis/analysis_R.qmd`](Analysis/analysis_R.qmd) | Confirmatory R/Stan analysis, tables, and figures |
| [`Analysis/analysis_python.py`](Analysis/analysis_python.py) | Reusable Python/PyMC implementation |
| [`Analysis/analysis_Py.ipynb`](Analysis/analysis_Py.ipynb) | Concise notebook interface to the Python implementation |
| [`Analysis/README.md`](Analysis/README.md) | Execution, outputs, and interpretation notes |
| [`renv.lock`](renv.lock) | R dependency snapshot |
| [`requirements.txt`](requirements.txt) | Direct Python dependencies |

Model outputs, rendered reports, figures, caches, and local data are ignored. They are regenerated under `outputs/` when the analyses run.

## Reproduce

### R (published workflow)

Requirements: R, Quarto, a working C++ toolchain, and CmdStan.

```r
install.packages("renv")
renv::restore()
cmdstanr::install_cmdstan()
```

```bash
quarto render Analysis/analysis_R.qmd
```

The analysis fits eight nonlinear beta-regression models (two in Experiment 1 and six in Experiment 2), each with four chains and 4,000 iterations. A full run is computationally intensive. Cached model objects may be retained locally in `outputs/models/`.

### Python (independent implementation)

Requirements: Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Analysis/analysis_python.py --data-dir data
```

Use `--quick` only for a short software check; quick-run estimates are not suitable for inference.

## Statistical specification

For Experiment 1, relative subjective value is modeled as

$$
V = \frac{1}{(1 + bX)^s},
$$

where $X$ is the odds against receipt of the probabilistic reinforcer, $b$ is the discounting-rate parameter, and $s$ is a scaling parameter. Experiment 2 uses the generalized form

\frac{A_{\mathrm{smaller}}}{A_{\mathrm{larger}}}
=
\left(\frac{1+bX_{\mathrm{smaller}}}{1+bX_{\mathrm{larger}}}\right)^s.

The published Bayesian curve-fitting analyses used beta likelihoods, positive Normal(1, 10) priors for $b$ and $s$, four chains, 2,000 warmup iterations, and 2,000 retained iterations per chain. Area under the empirical curve (AUC) provides a model-free summary of discounting; the confirmatory R analysis tests amount effects with mixed-effects beta regression. The article reports R 4.2.1; `renv.lock` records the later R 4.4.1 replication environment maintained by this repository.

## Transparency and scope

- Experimental replication conditions are plotted but excluded from inferential analyses, following the article.
- Group means supplied in the OSF files are retained for descriptive curve fitting but excluded from subject-level amount-effect models.
- The repository does not contain private drafts, correspondence, Qualtrics exports, receipts, literature libraries, or research data.
- Software versions may affect Monte Carlo estimates. Report seeds, package versions, convergence diagnostics, and any deviations from the published specification.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) to cite the article and this software. Code is released under the [MIT License](LICENSE). Data use and attribution are governed by the terms stated on OSF.

## Suggested GitHub metadata

**Description:** Reproducible R and Python analyses of probability discounting in pigeons using Bayesian hyperboloid models and open OSF data.

**Topics:** `probability-discounting`, `decision-making`, `behavior-analysis`, `comparative-cognition`, `pigeons`, `bayesian-analysis`, `brms`, `pymc`, `open-science`, `reproducible-research`
