# Copyright (c) 2023 Darren Erik Vengroff
"""Command-line interface for impact charts."""

import re
import sys
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from logargparser import LoggingArgumentParser
from sklearn.linear_model import LinearRegression

from impactchart.model import XGBoostImpactModel

logger = getLogger(__name__)


def linreg(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_col: str,
    w_col: Optional[str] = None,
) -> Dict[str, Any]:
    regressor = LinearRegression()

    logger.info("Fitting linear regression.")
    logger.info(f"Shape of df: {df.shape}")

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


def _split_on_colon(variable_name: str) -> Tuple[str, str]:
    """
    Split a variable name on a colon, if present.

    Parameters
    ----------
    variable_name
        Name of the variable to split.
    Returns
    -------
        tuple of name and human readable string from after the colon if present.
    """
    var_and_name = variable_name.split(":")
    return var_and_name[0], (
        var_and_name[1] if len(var_and_name) > 1 else var_and_name[0]
    )


def optimize(args):
    """
    Implements the optimize command.

    Parameters
    ----------
    args
        Arguments parsed from the command line.

    Returns
    -------
        None
    """
    data_path = Path(args.data)
    output_path = Path(args.output)

    # Drop any label after a ':'.
    y_col, y_name = _split_on_colon(args.y_column)

    df = read_and_filter_data(data_path, y_col, args.filter)

    x_col_args = args.X_columns
    x_cols, x_names = [
        list(t) for t in zip(*[_split_on_colon(col) for col in x_col_args])
    ]

    # Weigh by weight column.
    w_col, w_name = (
        _split_on_colon(args.w_column) if args.w_column is not None else (None, None)
    )

    if not args.dry_run:
        seed = int(args.seed, 0)

        impact_model = XGBoostImpactModel(random_state=seed)

        if w_col is not None:
            sample_weight = df[w_col]
        else:
            sample_weight = None

        xgb_params = impact_model.optimize_hyperparameters(
            df[x_cols],
            df[y_col],
            sample_weight,
            optimization_scoring_metric=args.scoring,
        )

        # Some of these need to be converted from np
        # data types so they serialize to yaml nicely.
        for key in ["learning_rate", "subsample"]:
            xgb_params[key] = float(xgb_params[key])

        logger.info(f"Writing to output file `{output_path}`")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"All X shape: {df.shape}")
        df = df.dropna(subset=x_cols + [y_col])
        if w_col is not None:
            df = df.dropna(subset=[w_col])
        logger.info(f"Dropna X shape: {df.shape}")

        linreg_params = linreg(df, x_cols, y_col, w_col)

        features = [{col: name} for col, name in zip(x_cols, x_names)]
        target = {y_col: y_name}

        params = {
            "features": features,
            "target": target,
            "linreg": linreg_params,
            "xgb": {
                "params": xgb_params,
                "target": impact_model.best_score_,
                "score": impact_model.r2_,
            },
        }

        if w_col is not None:
            params["weight"] = {w_col: w_name}

        with open(output_path, "w") as f:
            yaml.dump(params, f, sort_keys=True)


def read_and_filter_data(
    data_path, y_col: str, filters: Optional[Iterable[str]] = None
):
    """Read and filter the data."""
    if filters is None:
        filters = []

    filter_pattern = re.compile(r"^(\w+)\s*(<|>|<=|>=|=|==|===|!=)\s*(\w+)$")

    filter_expressions = []

    for f in filters:
        match = filter_pattern.match(f)
        if match is not None:
            logger.info(
                f"Parsed filter {match.group(1)} {match.group(2)} {match.group(3)}"
            )
            filter_expressions.append(
                {
                    "column": match.group(1),
                    "operator": match.group(2),
                    "value": match.group(3),
                }
            )

    string_filter_names = [
        f["column"] for f in filter_expressions if f["operator"] == "==="
    ]

    str_col_types = {
        # TODO - get a definitive list of geo columns that are strings from
        # a new API to be added to censusdis.
        col: str
        for col in set(
            ["STATE", "COUNTY", "TRACT", "BLOCK_GROUP"] + string_filter_names
        )
    }

    df = pd.read_csv(data_path, header=0, dtype=str_col_types)

    logger.info(f"Initial rows: {len(df.index)}")

    for f in filter_expressions:
        col = f["column"]
        op = f["operator"]
        value = f["value"]

        # Before we can do the comparison, we have to
        # cast the value that originally came in as a
        # string on the command line to the type of the
        # column we are comparing it to.
        typed_value = df[col].dtypes.type(value)

        logger.info(f"Filtering {col} {op} {typed_value}")

        exprs = {
            "=": df[col] == typed_value,
            "==": df[col] == typed_value,
            "===": df[col] == typed_value,
            "!=": df[col] != typed_value,
            "<": df[col] < typed_value,
            ">": df[col] > typed_value,
            "<=": df[col] <= typed_value,
            ">=": df[col] >= typed_value,
        }

        df = df[exprs[op]]

        logger.info(f"Remaining rows: {len(df.index)}")

    df = df.dropna(subset=[y_col])
    logger.info(f"Shape after dropna: {df.shape}")
    if len(df.index) == 0:
        logger.warning(f"After removing nan from {y_col}, no data is left.")
        sys.exit(1)
    logger.info(
        f"Range: {df[y_col].min()} - {df[y_col].max()}; mean: {df[y_col].mean()}"
    )

    return df


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
    *,
    plot_linreg: bool = False,
    linreg_coefs: Optional[Iterable[float]] = None,
    linreg_intercept: Optional[float] = None,
    feature_names: Optional[Mapping[str, str]] = None,
    y_name: Optional[str] = None,
    subtitle: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    yformatter: str = "comman",
    x_formatter_default: str = "comma",
    x_formatters: Optional[Dict[str, str]] = None,
):
    if plot_linreg:
        reg_linreg = _linreg_from_coefficients(linreg_coefs, linreg_intercept)
    else:
        reg_linreg = None

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
        y_formatter=yformatter,
        x_formatters=x_formatters,
        x_formatter_default=x_formatter_default,
    )
    logger.info("Fitting complete.")

    if filename_prefix is None:
        filename_prefix = ""
    if filename_suffix is None:
        filename_suffix = ""

    for feature, (fig, ax) in impact_charts.items():
        logger.info(f"Plotting {feature}")

        # Plot the linear line by plotting the output of the linear
        # model with all other features zeroed out so they have no
        # effect.
        df_one_feature = pd.DataFrame(
            {f: X[f] if f == feature else 0.0 for f in X.columns}
        )

        if reg_linreg is not None:
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

        if xmin is not None or ymin is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None or xmin is not None:
            ax.set_ylim(ymin, ymax)

        logger.info(f"Saving impact chart for {feature}.")
        fig.savefig(output_path / f"{filename_prefix}{feature}{filename_suffix}.png")


def plot(args: Namespace) -> None:
    """
    Implements the plot command.

    Parameters
    ----------
    args
        Arguments parsed from the command line.

    Returns
    -------
        None
    """
    with open(args.parameters) as f:
        param_file_contents = yaml.full_load(f)

    y_col, y_name = list(param_file_contents["target"].items())[0]

    feature_names = param_file_contents["features"]
    x_cols = [list(feature_dict.keys())[0] for feature_dict in feature_names]
    feature_dict = {}
    for d in feature_names:
        feature_dict |= d

    data_path = Path(args.data)

    df = read_and_filter_data(data_path, y_col, args.filter)

    xgb_params = param_file_contents["xgb"]["params"]

    if args.linreg and "linreg" in param_file_contents:
        linreg_coefs = param_file_contents["linreg"]["coefficients"]
        linreg_intercept = param_file_contents["linreg"]["intercept"]
    else:
        linreg_coefs = None
        linreg_intercept = None

    X = df[x_cols]
    y = df[y_col]

    if "weight" in param_file_contents:
        w_col = list(param_file_contents["weight"].keys())[0]
        w = df[w_col]
    else:
        w = None

    k = args.k
    seed = int(args.seed, 0)

    impact_model = XGBoostImpactModel(
        ensemble_size=k,
        random_state=seed,
        estimator_kwargs=xgb_params,
        optimize_hyperparameters=False,
    )
    impact_model.fit(X, y, sample_weight=w)

    output_path = Path(args.output)

    suffix = None

    xmin = args.xmin
    xmax = args.xmax
    ymin = args.ymin
    ymax = args.ymax

    if args.xformat is not None:
        x_formatters = {feature: x_format for feature, x_format in args.xformat}
    else:
        x_formatters = {}

    plot_impact_charts(
        impact_model,
        X,
        output_path,
        plot_linreg=args.linreg,
        linreg_coefs=linreg_coefs,
        linreg_intercept=linreg_intercept,
        feature_names=feature_dict,
        y_name=y_name,
        subtitle=args.subtitle,
        filename_suffix=suffix,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        yformatter=args.yformat.lower(),
        x_formatters=x_formatters,
        x_formatter_default=args.xformat_default,
    )


def add_data_arguments(parser) -> None:
    parser.add_argument("-f", "--filter", type=str, nargs="*")

    parser.add_argument("data", help="Input data file.")


def main():
    parser = LoggingArgumentParser(logger, prog="impactchart")

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
        "--scoring",
        type=str,
        help="Scoring method to use when optimizing."
        "See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter",
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

    plot_parser.add_argument("--xmin", type=float, help="Min value on the x axis.")
    plot_parser.add_argument("--xmax", type=float, help="Max value on the x axis.")

    plot_parser.add_argument("--ymin", type=float, help="Min value on the y axis.")
    plot_parser.add_argument("--ymax", type=float, help="Max value on the y axis.")

    format_choices = [
        "comma",
        "dollar",
        "percentage",
    ]

    plot_parser.add_argument(
        "--yformat",
        type=str,
        choices=format_choices,
        default="comma",
        help="How to format the Y ticks.",
    )

    def xformat_tuple_type(x_format_arg: str) -> Tuple[str, str]:
        """Parse a xformat arg value into a feature and format tuple."""
        if x_format_arg.count(":") != 1:
            raise ValueError(
                "Invalid xformat '{}'. Must contain exactly one ':'".format(
                    x_format_arg
                )
            )
        feature, x_format = x_format_arg.split(":")
        if x_format not in format_choices:
            raise ValueError(f"Invalid xformat {x_format} for feature '{feature}'.")
        return feature, x_format

    plot_parser.add_argument(
        "--xformat",
        type=xformat_tuple_type,
        nargs="*",
        help=f"Use value 'feature:format' to specify the format for each feature. format is one of {format_choices}.",
    )

    plot_parser.add_argument(
        "--xformat-default",
        type=str,
        choices=format_choices,
        default="comma",
        help="How to format the x ticks for features not specificed in --xformat.",
    )

    args = parser.parse_args()

    logger.info(f"Command is {args.command}")

    if args.command == "optimize":
        optimize(args)
    elif args.command == "plot":
        plot(args)


if __name__ == "__main__":
    main()
