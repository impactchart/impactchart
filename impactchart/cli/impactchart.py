# Copyright (c) 2023 Darren Erik Vengroff
"""Command-line interface for impact charts."""

import sys

from typing import Any, Dict, Iterable, Optional
from logging import getLogger

from argparse import ArgumentParser

from logargparser import LoggingArgumentParser

from pathlib import Path

import pandas as pd
import xgboost
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV


logger = getLogger(__name__)


def optimize_xgb(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_col: str,
    w_col: Optional[str] = None,
) -> Dict[str, Any]:
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
        random_state=17,
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
        xgb_params = optimize_xgb(df, x_cols, y_col, w_col=w_col)

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


def plot(args):
    pass


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

    plot_parser = subparsers.add_parser("plot", help="Generate an impact chart.")

    optimize_parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output yaml file."
    )

    optimize_parser.add_argument(
        "-X",
        "--X-columns",
        type=str,
        nargs="+",
        required=True,
        help="X columns (features) for fitting the models.",
    )

    optimize_parser.add_argument(
        "-y",
        "--y-column",
        type=str,
        required=True,
        help="What variable are we trying to predict?",
    )

    optimize_parser.add_argument(
        "-w",
        "--w-column",
        type=str,
        help="Weight column.",
    )

    optimize_parser.add_argument(
        "data", help="Input data file. Typically from select.py."
    )

    plot_parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output plot file."
    )

    args = parser.parse_args()

    logger.info(f"Command is {args.command}")

    if args.command == "optimize":
        optimize(args)
    elif args.command == "plot":
        plot(args)


if __name__ == "__main__":
    main()
