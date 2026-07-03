"""Independent PyMC implementation of Oliveira et al. (2025).

The confirmatory numerical replication is Analysis/analysis_R.qmd. This module
uses the same data transformations, hyperboloid equations, positive Normal(1,
10) priors for b and s, and sampling schedule. Its Bayesian AUC model is a
companion analysis rather than a reproduction of the article's frequentist
mixed-effects beta regression.

Data: https://osf.io/scwg3/
Article: https://doi.org/10.1002/jeab.70042
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

SEED = 20250714
SUBJECT_LEVELS = [f"P{i}" for i in range(41, 49)] + ["Mean"]
REQUIRED_EXP1 = {"Subject", "Amount", "Probability", "SV"}
REQUIRED_EXP2 = {
    "Subject",
    "Amount",
    "Multiplier",
    "Probability_Sm",
    "Probability_Lg",
    "SV",
}


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a readable error when an OSF file has an unexpected schema."""
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _require_beta_outcome(data: pd.DataFrame, label: str) -> None:
    """Validate rather than silently alter values used in beta regression."""
    values = data["RSV"].to_numpy()
    if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise ValueError(f"{label} contains RSV values outside the open interval (0, 1).")


def load_study_data(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load and validate the two OSF CSV files.

    Replication conditions are retained and marked by ``Rep``. OSF-provided
    ``Mean`` rows are retained but are never treated as independent pigeons.
    """
    data_dir = Path(data_dir)
    path_exp1 = data_dir / "PigeonPD_Exp1.csv"
    path_exp2 = data_dir / "PigeonPD_Exp2.csv"
    if not path_exp1.exists() or not path_exp2.exists():
        raise FileNotFoundError(
            "Download PigeonPD_Exp1.csv and PigeonPD_Exp2.csv from "
            "https://osf.io/scwg3/ and place them in data/."
        )

    raw_exp1 = pd.read_csv(path_exp1, dtype={"Probability": "string"})
    raw_exp2 = pd.read_csv(path_exp2, dtype={"Multiplier": "string"})
    _require_columns(raw_exp1, REQUIRED_EXP1, "Experiment 1")
    _require_columns(raw_exp2, REQUIRED_EXP2, "Experiment 2")

    probability = raw_exp1["Probability"].str.replace(r"R$", "", regex=True).astype(float) / 100
    exp1 = pd.DataFrame(
        {
            "Subject": raw_exp1["Subject"],
            "Experiment": 1,
            "Amount": raw_exp1["Amount"].astype(int),
            "Multiplier": 1.0,
            "Rep": raw_exp1["Probability"].str.endswith("R"),
            "Probability_Sm": 1.0,
            "Probability_Lg": probability,
            "OAs_Sm": 0.0,
            "OAs_Lg": (1 - probability) / probability,
            "RSV": raw_exp1["SV"] / raw_exp1["Amount"],
        }
    )

    multiplier = raw_exp2["Multiplier"].str.replace(r"R$", "", regex=True).astype(float)
    probability_sm = raw_exp2["Probability_Sm"].astype(float) / 100
    probability_lg = raw_exp2["Probability_Lg"].astype(float) / 100
    exp2 = pd.DataFrame(
        {
            "Subject": raw_exp2["Subject"],
            "Experiment": 2,
            "Amount": raw_exp2["Amount"].astype(int),
            "Multiplier": multiplier,
            "Rep": raw_exp2["Multiplier"].str.endswith("R"),
            "Probability_Sm": probability_sm,
            "Probability_Lg": probability_lg,
            "OAs_Sm": (1 - probability_sm) / probability_sm,
            "OAs_Lg": (1 - probability_lg) / probability_lg,
            "RSV": raw_exp2["SV"] / raw_exp2["Amount"],
        }
    )

    _require_beta_outcome(exp1, "Experiment 1")
    _require_beta_outcome(exp2, "Experiment 2")
    data = pd.concat([exp1, exp2], ignore_index=True)
    unexpected = sorted(set(data["Subject"]).difference(SUBJECT_LEVELS))
    if unexpected:
        raise ValueError(f"Unexpected subject identifiers: {unexpected}")
    data["Subject"] = pd.Categorical(data["Subject"], SUBJECT_LEVELS, ordered=True)
    return data.sort_values(
        ["Experiment", "Subject", "Amount", "Multiplier", "OAs_Lg"],
        ignore_index=True,
    )


def trapezoid_auc(x: Iterable[float], y: Iterable[float]) -> float:
    """Empirical AUC after within-condition odds-against normalization.

    No unobserved anchor is inserted, matching the published computation.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    order = np.argsort(x_values)
    x_values = x_values[order] / x_values[order].max()
    y_values = y_values[order]
    return float(np.trapezoid(y_values, x_values))


def compute_auc(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate empirical AUC for every nonreplication condition."""
    columns = ["Subject", "Experiment", "Amount", "Multiplier"]
    auc = (
        data.loc[~data["Rep"]]
        .groupby(columns, observed=True)
        .apply(lambda group: trapezoid_auc(group["OAs_Lg"], group["RSV"]), include_groups=False)
        .rename("AUC")
        .reset_index()
    )
    check = auc.query("Subject == 'P41' and Experiment == 1 and Amount == 16")["AUC"].iloc[0]
    if not np.isclose(check, 0.125, atol=0.001):
        raise RuntimeError(f"AUC validation failed: expected .125, obtained {check:.6f}")
    return auc


def hyperboloid_exp1(oas_lg: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Equation 1: certain versus probabilistic reinforcement."""
    return 1 / (1 + b * oas_lg) ** s


def hyperboloid_exp2(
    oas_sm: np.ndarray,
    oas_lg: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Equation 2: two probabilistic reinforcers."""
    return ((1 + b * oas_sm) / (1 + b * oas_lg)) ** s


def fit_discount_model(
    data: pd.DataFrame,
    experiment: int,
    *,
    draws: int = 2_000,
    tune: int = 2_000,
    chains: int = 4,
    cores: int = 4,
    seed: int = SEED,
) -> az.InferenceData:
    """Fit one subject-specific nonlinear beta-regression model.

    ``b`` and ``s`` use the article's positive Normal(1, 10) priors. The
    article does not report a precision prior; this implementation states its
    HalfNormal(10) prior explicitly. Parameters are subject-specific and are
    not partially pooled.
    """
    if experiment not in (1, 2):
        raise ValueError("experiment must be 1 or 2")
    model_data = data.loc[~data["Rep"]].copy()
    subjects = [str(item) for item in model_data["Subject"].cat.remove_unused_categories().cat.categories]
    model_data["Subject"] = pd.Categorical(model_data["Subject"], subjects, ordered=True)
    subject_index = model_data["Subject"].cat.codes.to_numpy()

    with pm.Model(coords={"subject": subjects, "observation": np.arange(len(model_data))}):
        index = pm.Data("subject_index", subject_index, dims="observation")
        oas_lg = pm.Data("oas_lg", model_data["OAs_Lg"].to_numpy(), dims="observation")
        b = pm.TruncatedNormal("b", mu=1, sigma=10, lower=0, dims="subject")
        s = pm.TruncatedNormal("s", mu=1, sigma=10, lower=0, dims="subject")
        phi = pm.HalfNormal("phi", sigma=10, dims="subject")

        if experiment == 1:
            mu = 1 / (1 + b[index] * oas_lg) ** s[index]
        else:
            oas_sm = pm.Data("oas_sm", model_data["OAs_Sm"].to_numpy(), dims="observation")
            mu = ((1 + b[index] * oas_sm) / (1 + b[index] * oas_lg)) ** s[index]

        pm.Beta(
            "RSV",
            alpha=mu * phi[index],
            beta=(1 - mu) * phi[index],
            observed=model_data["RSV"].to_numpy(),
            dims="observation",
        )
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            target_accept=0.99,
            nuts={"max_treedepth": 15},
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )


def fit_auc_amount_model(
    auc_data: pd.DataFrame,
    *,
    draws: int = 2_000,
    tune: int = 2_000,
    chains: int = 4,
    cores: int = 4,
    seed: int = SEED,
) -> az.InferenceData:
    """Bayesian mixed-effects beta regression for the 32-vs-16 amount effect."""
    data = auc_data.loc[auc_data["Subject"] != "Mean"].copy()
    subject = pd.Categorical(data["Subject"])
    amount_32 = (data["Amount"].to_numpy() == 32).astype(float)

    with pm.Model(coords={"subject": subject.categories, "observation": np.arange(len(data))}):
        subject_index = pm.Data("subject_index", subject.codes, dims="observation")
        amount = pm.Data("amount_32", amount_32, dims="observation")
        intercept = pm.Normal("intercept", 0, 2.5)
        amount_effect = pm.Normal("amount_effect", 0, 1)
        subject_sd = pm.HalfNormal("subject_sd", 1)
        subject_offset = pm.Normal("subject_offset", 0, 1, dims="subject")
        phi = pm.Exponential("phi", 1)
        eta = intercept + amount_effect * amount + subject_sd * subject_offset[subject_index]
        mu = pm.math.sigmoid(eta)
        pm.Beta(
            "AUC",
            alpha=mu * phi,
            beta=(1 - mu) * phi,
            observed=data["AUC"].to_numpy(),
            dims="observation",
        )
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            target_accept=0.95,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )


def _fit_or_load(path: Path, fit_function, *args, **kwargs) -> az.InferenceData:
    if path.exists():
        return az.from_netcdf(path)
    result = fit_function(*args, **kwargs)
    result.to_netcdf(path)
    return result


def _check_diagnostics(summary: pd.DataFrame, label: str) -> None:
    if "r_hat" in summary and summary["r_hat"].notna().any():
        if summary["r_hat"].max() > 1.01:
            warnings.warn(f"{label}: at least one R-hat exceeds 1.01", stacklevel=2)
    if "ess_bulk" in summary and summary["ess_bulk"].min() < 400:
        warnings.warn(f"{label}: at least one bulk ESS is below 400", stacklevel=2)


def _plot_auc_differences(auc_data: pd.DataFrame, output_path: Path) -> None:
    individual = auc_data.loc[auc_data["Subject"] != "Mean"].copy()
    wide = individual.pivot(
        index=["Subject", "Experiment", "Multiplier"], columns="Amount", values="AUC"
    ).reset_index()
    wide["Difference"] = wide[16] - wide[32]
    wide["Condition"] = np.where(
        wide["Experiment"] == 1,
        "Experiment 1",
        "Experiment 2: multiplier " + wide["Multiplier"].astype(str),
    )
    sns.set_theme(style="ticks", context="notebook")
    graph = sns.catplot(
        data=wide,
        x="Subject",
        y="Difference",
        col="Condition",
        col_wrap=2,
        kind="point",
        color="#2C6E49",
        height=3.2,
        aspect=1.25,
    )
    for axis in graph.axes.flat:
        axis.axhline(0, color="0.4", linestyle="--", linewidth=1)
    graph.set_axis_labels("Pigeon", "AUC difference (16 - 32 pellets)")
    graph.tight_layout()
    graph.savefig(output_path, dpi=300)
    plt.close(graph.figure)


def run_analysis(
    data_dir: Path,
    output_dir: Path,
    *,
    quick: bool = False,
    skip_models: bool = False,
) -> None:
    """Validate data, compute AUC, fit models, and write reproducible outputs."""
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models" / "python"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_study_data(data_dir)
    auc_data = compute_auc(data)
    data.to_csv(table_dir / "processed_data_python.csv", index=False)
    auc_data.to_csv(table_dir / "empirical_auc_python.csv", index=False)
    _plot_auc_differences(auc_data, figure_dir / "auc_amount_differences_python.png")

    if skip_models:
        return

    if quick:
        warnings.warn(
            "Quick mode is a software check only; do not use its estimates for inference.",
            stacklevel=2,
        )
        sampling = {"draws": 100, "tune": 100, "chains": 2, "cores": 2}
    else:
        sampling = {"draws": 2_000, "tune": 2_000, "chains": 4, "cores": 4}

    summaries: list[pd.DataFrame] = []
    confirmatory = data.loc[~data["Rep"]]
    curve_conditions = [
        (1, amount, 1.0) for amount in (16, 32)
    ] + [
        (2, amount, multiplier)
        for amount in (16, 32)
        for multiplier in (1.0, 0.75, 0.25)
    ]
    for experiment, amount, multiplier in curve_conditions:
        condition = confirmatory.loc[
            (confirmatory["Experiment"] == experiment)
            & (confirmatory["Amount"] == amount)
            & np.isclose(confirmatory["Multiplier"], multiplier)
        ].copy()
        label = f"exp{experiment}_{amount}_m{multiplier:g}"
        idata = _fit_or_load(
            model_dir / f"{label}.nc",
            fit_discount_model,
            condition,
            experiment,
            **sampling,
        )
        summary = az.summary(idata, var_names=["b", "s", "phi"]).reset_index(names="Parameter")
        summary.insert(0, "Model", label)
        _check_diagnostics(summary, label)
        summaries.append(summary)

    pd.concat(summaries, ignore_index=True).to_csv(
        table_dir / "posterior_parameters_python.csv", index=False
    )

    auc_summaries: list[pd.DataFrame] = []
    auc_conditions = [(1, 1.0)] + [(2, item) for item in (1.0, 0.75, 0.25)]
    for experiment, multiplier in auc_conditions:
        condition = auc_data.loc[
            (auc_data["Experiment"] == experiment)
            & np.isclose(auc_data["Multiplier"], multiplier)
        ]
        label = f"auc_exp{experiment}_m{multiplier:g}"
        idata = _fit_or_load(
            model_dir / f"{label}.nc",
            fit_auc_amount_model,
            condition,
            **sampling,
        )
        summary = az.summary(idata, var_names=["amount_effect"], hdi_prob=0.95).reset_index(
            names="Parameter"
        )
        summary.insert(0, "Model", label)
        _check_diagnostics(summary, label)
        auc_summaries.append(summary)

    pd.concat(auc_summaries, ignore_index=True).to_csv(
        table_dir / "auc_amount_effects_python.csv", index=False
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 2 x 100-draw software checks; estimates are not inferential.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Validate and process data without sampling Bayesian models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_analysis(
        arguments.data_dir,
        arguments.output_dir,
        quick=arguments.quick,
        skip_models=arguments.skip_models,
    )
