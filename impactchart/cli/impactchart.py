# Copyright (c) 2023 Darren Erik Vengroff
"""Command-line interface for impact charts."""

import sys

from typing import Any, Dict, Iterable, Mapping, Optional
from logging import getLogger

from argparse import ArgumentParser

from logargparser import LoggingArgumentParser

from pathlib import Path

import pandas as pd
import numpy as np

from matplotlib.ticker import FuncFormatter, PercentFormatter

import xgboost
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

from impactchart.model import XGBoostImpactModel


logger = getLogger(__name__)


def optimize_xgb(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_col: str,
    w_col: Optional[str] = None,
    random_state: Optional[int] = 17,
) -> Dict[str, Any]:
    logger.info("Optimizing the XBG model.")

    reg_xgb = xgboost.XGBRegressor()

    param_dist = {
        "n_estimators": stats.randint(10, 100),
        "learning_rate": stats.uniform(0.01, 0.07),
        "subsample": stats.uniform(0.3, 0.7),
        "max_depth": stats.randint(2, 6),
        "min_child_weight": stats.randint(1, 4),
    }

    reg = RandomizedSearchCV(
        reg_xgb,
        param_distributions=param_dist,
        n_iter=200,
        error_score=0,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )

    X = df[list(x_cols)]
    y = df[y_col]

    if w_col is not None:
        w = df[w_col]
    else:
        w = None

    reg.fit(X, y, sample_weight=w)

    result = {
        "params": reg.best_params_,
        "target": float(reg.best_score_),
        "score": float(reg.best_estimator_.score(X, y, sample_weight=w)),
    }

    result["params"]["learning_rate"] = float(result["params"]["learning_rate"])
    result["params"]["subsample"] = float(result["params"]["subsample"])

    logger.info("Optimization complete.")

    return result


def linreg(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_col: str,
    w_col: Optional[str] = None,
) -> Dict[str, Any]:
    regressor = LinearRegression()

    if w_col is None:
        model = regressor.fit(df[x_cols], df[y_col])
        score = regressor.score(df[x_cols], df[y_col])
    else:
        model = regressor.fit(df[x_cols], df[y_col], sample_weight=df[w_col])
        score = regressor.score(df[x_cols], df[y_col], sample_weight=df[w_col])

    coefficients = model.coef_.tolist()
    intercept = model.intercept_

    return {
        "coefficients": coefficients,
        "intercept": float(intercept),
        "score": float(score),
    }


def optimize(args):
    data_path = Path(args.data)
    output_path = Path(args.output)
    df = pd.read_csv(
        data_path, header=0, dtype={"STATE": str, "COUNTY": str, "TRACT": str}
    )
    x_cols = args.X_columns
    y_col = args.y_column
    # Weigh by total renters.
    w_col = args.w_column
    logger.info(f"Input shape: {df.shape}")
    df = df.dropna(subset=[y_col])
    logger.info(f"Shape after dropna: {df.shape}")
    if len(df.index) == 0:
        logger.warning(f"After removing nan from {y_col}, no data is left.")
        sys.exit(1)
    logger.info(
        f"Range: {df[y_col].min()} - {df[y_col].max()}; mean: {df[y_col].mean()}"
    )
    if not args.dry_run:
        xgb_params = optimize_xgb(
            df, x_cols, y_col, w_col=w_col, random_state=args.seed
        )

        logger.info(f"Writing to output file `{output_path}`")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"All X shape: {df.shape}")
        df = df.dropna(subset=x_cols)
        logger.info(f"Dropna X shape: {df.shape}")

        linreg_params = linreg(df, x_cols, y_col)

        params = {
            "linreg": linreg_params,
            "xgb": xgb_params,
        }
        with open(output_path, "w") as f:
            yaml.dump(params, f, sort_keys=True)


def _linreg_from_coefficients(coef, intercept):
    reg_linreg = LinearRegression()
    # Instead of fitting, we are just going to kludge in
    # the known coefficients.
    reg_linreg.coef_ = np.array(coef)
    reg_linreg.intercept_ = intercept
    return reg_linreg


def _plot_id(feature, k, n, seed):
    return f"(f = {feature}; n = {n:,.0f}; k = {k}; s = {seed:08X})"


def plot_impact_charts(
    impact_model: XGBoostImpactModel,
    X: pd.DataFrame,
    output_path: Path,
    k: int,
    seed: int,
    *,
    linreg: bool = False,
    linreg_coefs: Optional[Iterable[float]] = None,
    linreg_intercept: Optional[float] = None,
    feature_names: Optional[Mapping[str, str]] = None,
    y_name: Optional[str] = None,
    subtitle: Optional[str] = None,
):
    if linreg:
        reg_linreg = _linreg_from_coefficients(linreg_coefs, linreg_intercept)

    logger.info("Fitting the impact chart model.")
    impact_charts = impact_model.impact_charts(
        X,
        X.columns,
        subplots_kwargs=dict(
            figsize=(12, 8),
        ),
        feature_names=feature_names,
        y_name=y_name,
        subtitle=subtitle,
    )
    logger.info("Fitting complete.")

    dollar_formatter = FuncFormatter(
        lambda d, pos: f"\\${d:,.0f}" if d >= 0 else f"(\\${-d:,.0f})"
    )

    for feature, (fig, ax) in impact_charts.items():
        logger.info(f"Plotting {feature}")

        # Plot the linear line by plotting the output of the linear
        # model with all other features zeroed out so they have no
        # effect.
        df_one_feature = pd.DataFrame(
            {f: X[f] if f == feature else 0.0 for f in X.columns}
        )

        if linreg:
            df_one_feature["impact"] = reg_linreg.predict(df_one_feature)

            df_endpoints = pd.concat(
                [
                    df_one_feature.nsmallest(1, feature),
                    df_one_feature.nlargest(1, feature),
                ]
            )

            ax = df_endpoints.plot.line(
                feature, "impact", color="orange", ax=ax, label="Linear Model"
            )

        plot_id = _plot_id(feature, k, len(X.index), seed)
        ax.text(
            0.99,
            0.01,
            plot_id,
            fontsize=8,
            backgroundcolor="white",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

        col_is_fractional = feature.startswith("frac_")

        if col_is_fractional:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_xlim(-0.05, 1.05)
        else:
            ax.xaxis.set_major_formatter(dollar_formatter)
            ax.set_xlim(-5_000, max(10_000, df_one_feature[feature].max()) * 1.05)

        ax.grid(visible=True)

        logger.info(f"Saving impact chart for {feature}.")
        fig.savefig(output_path / f"{feature}.png")


def plot(args):
    data_path = Path(args.data)

    df = pd.read_csv(
        data_path, header=0, dtype={"STATE": str, "COUNTY": str, "TRACT": str}
    )

    with open(args.parameters) as f:
        param_file_contents = yaml.full_load(f)

    xgb_params = param_file_contents["xgb"]["params"]

    if args.linreg and "linreg" in param_file_contents:
        linreg_coefs = param_file_contents["linreg"]["coefficients"]
        linreg_intercept = param_file_contents["linreg"]["intercept"]
    else:
        linreg_coefs = None
        linreg_intercept = None

    x_col_args = args.X_columns

    x_cols = [col.split(":")[0] for col in x_col_args]

    feature_names = {}
    for col in x_col_args:
        col_and_name = col.split(":")
        if len(col_and_name) > 1:
            feature_names[col_and_name[0]] = col_and_name[1]

    X = df[x_cols]
    y = df[args.y_column]
    w = df[args.w_column]

    k = args.k
    seed = int(args.seed, 0)

    impact_model = XGBoostImpactModel(
        ensemble_size=k, random_state=seed, estimator_kwargs=xgb_params
    )
    impact_model.fit(X, y, sample_weight=w)

    output_path = Path(args.output)

    plot_impact_charts(
        impact_model,
        X,
        output_path,
        k,
        seed,
        linreg=args.linreg,
        linreg_coefs=linreg_coefs,
        linreg_intercept=linreg_intercept,
        feature_names=feature_names,
        y_name=args.y_name,
        subtitle=args.subtitle,
    )


def add_data_arguments(parser) -> None:
    parser.add_argument(
        "-X",
        "--X-columns",
        type=str,
        nargs="+",
        required=True,
        help="X columns (features) for fitting the models.",
    )

    parser.add_argument(
        "-y",
        "--y-column",
        type=str,
        required=True,
        help="What variable are we trying to predict?",
    )

    parser.add_argument(
        "-w",
        "--w-column",
        type=str,
        help="Weight column.",
    )

    parser.add_argument("data", help="Input data file. Typically from select.py.")


def main():
    parser = LoggingArgumentParser(logger, prog="censusdis")

    subparsers = parser.add_subparsers(
        parser_class=ArgumentParser,
        required=True,
        dest="command",
        description="Choose one of the following commands.",
    )

    parser.add_argument("--dry-run", action="store_true")

    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize hyperparameters for a given data set."
    )

    optimize_parser.add_argument("-S", "--seed", type=str, default="17")

    plot_parser = subparsers.add_parser("plot", help="Generate an impact chart.")

    add_data_arguments(optimize_parser)
    add_data_arguments(plot_parser)

    optimize_parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output yaml file."
    )

    plot_parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output plot file."
    )

    plot_parser.add_argument(
        "-p",
        "--parameters",
        required=True,
        type=str,
        help="Model parameters (from optimize.py).",
    )

    plot_parser.add_argument("--linreg", action="store_true")

    plot_parser.add_argument("--subtitle", type=str, help="Subtitle for the plot.")
    plot_parser.add_argument("--y-name", type=str, help="Name for the y axis.")

    plot_parser.add_argument("-k", type=int, default=50)
    plot_parser.add_argument("-S", "--seed", type=str, default="0x3423CDF1")

    args = parser.parse_args()

    logger.info(f"Command is {args.command}")

    if args.command == "optimize":
        optimize(args)
    elif args.command == "plot":
        plot(args)


if __name__ == "__main__":
    main()
