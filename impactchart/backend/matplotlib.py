from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import Formatter, FuncFormatter, PercentFormatter

import textwrap

from ..backend.base import Backend


class MatplotlibBackend(Backend):
    def __init__(
        self,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        subplots_kwargs: Optional[Dict[str, Any]] = None,
        y_formatter: Optional[str] = None,
        x_formatter_default: Optional[str] = None,
        x_formatters: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        self._plot_kwargs = plot_kwargs or {}
        self._subplots_kwargs = subplots_kwargs or {}
        self._y_formatter = y_formatter
        self._x_formatter_default = x_formatter_default
        self._x_formatters = x_formatters or {}

    _dollar_formatter = FuncFormatter(
        lambda d, pos: f"\\${d:,.0f}" if d >= 0 else f"(\\${-d:,.0f})"
    )

    _percent_formatter = PercentFormatter(1.0, decimals=0)

    _comma_formatter = FuncFormatter(lambda d, pos: f"{d:,.0f}")

    _formatter_for_arg_value = {
        "percentage": _percent_formatter,
        "dollar": _dollar_formatter,
        "comma": _comma_formatter,
    }

    @classmethod
    def _axis_formatter(
        cls, formatter: Optional[Formatter | str] = None
    ) -> Formatter | None:
        if formatter is None:
            return None
        if isinstance(formatter, Formatter):
            return formatter
        else:
            return cls._formatter_for_arg_value[formatter]

    def plot(
            self,
            X: pd.DataFrame,
            df_impact: pd.DataFrame,
            feature: str,
            feature_name: str,
            *,
            min_impact: float,
            max_impact: float,
            marker_size: float,
            ensemble_marker_size: float,
            color: str,
            ensemble_color: str,
            y_name: Optional[str] = None,
            subtitle: Optional[str] = None,
            plot_id: Optional[str] = None,
    ):
        fig, ax = plt.subplots(**self._subplots_kwargs)

        ensemble_impact_label = "Impact of Individual Models"

        def _plot_for_ensemble_member(df_group):
            nonlocal ensemble_impact_label

            plot_x = X[feature]
            plot_y = df_group[feature]

            ax.plot(
                plot_x,
                plot_y,
                ".",
                markersize=ensemble_marker_size,
                color=ensemble_color,
                label=ensemble_impact_label,
                **self._plot_kwargs,
            )
            plot_kwargs = {}
            ensemble_impact_label = None

        df_impact.groupby("estimator")[["X_index", feature]].apply(
            _plot_for_ensemble_member
        )

        mean_impact = df_impact.groupby("X_index")[feature].mean()

        ax.plot(
            X[feature],
            mean_impact,
            ".",
            markersize=marker_size,
            color=color,
            label="Mean Impact",
        )

        # Do some basic labels and styling.
        ax.set_ylim(min_impact, max_impact)

        if y_name is not None:
            if subtitle is not None:
                title = f"Impact of {feature_name} on {y_name}\n{subtitle}"
            else:
                title = f"Impact of {feature_name} on {y_name}"
            ax.set_ylabel(f"Impact on {y_name}")
        else:
            if subtitle is not None:
                title = f"Impact of {feature_name} {subtitle}"
            else:
                title = f"Impact of {feature_name}"
            ax.set_ylabel("Impact")

        ax.set_title(textwrap.fill(title, width=80))
        ax.set_xlabel(feature_name)
        ax.grid()

        for handle in ax.legend().legend_handles:
            handle._sizes = [25]

        # Format the axes.
        y_axis_formatter = self._axis_formatter(self._y_formatter)
        if y_axis_formatter is not None:
            ax.yaxis.set_major_formatter(y_axis_formatter)
        if self._x_formatters is not None and feature in self._x_formatters:
            x_formatter = self._x_formatters[feature]
        else:
            x_formatter = self._x_formatter_default
        x_axis_formatter = self._axis_formatter(x_formatter)
        if x_axis_formatter is not None:
            ax.xaxis.set_major_formatter(x_axis_formatter)

        if plot_id is not None:
            ax.text(
                0.99,
                0.02,
                plot_id,
                fontsize=8,
                backgroundcolor="white",
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )

        return fig, ax
