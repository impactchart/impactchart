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
        """
        A backend class that used matplotlib to generate impact charts.

        Parameters
        ----------
        plot_kwargs
            Additional keyword args for matplotlib
        subplots_kwargs
            Additional kwargs for `plt.subplots` call to create the subplots.
        y_formatter
            How to format the y values. Can be one of the following: 'comma', 'percentage', 'dollar'.
        x_formatter_default:
            How to format the x axis unless uverridden for a particular feature by `x_formatters`.
            Same allowed values as `y_formatter`.
        x_formatters:
            A dictionary of how to format the x axis values. Keys are the features and values are the formats.
        """
        super().__init__()

        self._plot_kwargs = plot_kwargs or {}
        self._subplots_kwargs = subplots_kwargs or {}
        self._y_formatter = y_formatter
        self._x_formatter_default = x_formatter_default
        self._x_formatters = x_formatters or {}

        # These will get set up by begin().
        self._fig = None
        self._ax = None
        self._ensemble_impact_label = None

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

    def begin(
        self,
        *,
        feature: str,
        feature_name: str,
    ):
        self._fig, self._ax = plt.subplots(**self._subplots_kwargs)

        self._ensemble_impact_label = "Impact of Individual Models"

    def plot_ensemble_member_impact(
        self,
        x_feature: pd.Series,
        impact: pd.Series,
        index: int,
        *,
        feature_name: Optional[str] = None,
        ensemble_marker_size: float,
        ensemble_color: str,
    ):
        self._ax.plot(
            x_feature,
            impact,
            ".",
            markersize=ensemble_marker_size,
            color=ensemble_color,
            label=self._ensemble_impact_label,
            **self._plot_kwargs,
        )
        self._ensemble_impact_label = None

    def plot_mean_impact(
        self,
        x_feature: pd.Series,
        mean_impact: pd.Series,
        *,
        feature_name: Optional[str] = None,
        marker_size: float,
        color: str,
    ):
        self._ax.plot(
            x_feature,
            mean_impact,
            ".",
            markersize=marker_size,
            color=color,
            label="Mean Impact",
        )

    def end(
        self,
        *,
        feature: str,
        feature_name: str,
        min_impact: float,
        max_impact: float,
        y_name: Optional[str] = None,
        subtitle: Optional[str] = None,
        plot_id: Optional[str] = None,
    ):
        # Do some basic labels and styling.
        self._ax.set_ylim(min_impact, max_impact)

        if y_name is not None:
            if subtitle is not None:
                title = f"Impact of {feature_name} on {y_name}\n{subtitle}"
            else:
                title = f"Impact of {feature_name} on {y_name}"
            self._ax.set_ylabel(f"Impact on {y_name}")
        else:
            if subtitle is not None:
                title = f"Impact of {feature_name} {subtitle}"
            else:
                title = f"Impact of {feature_name}"
            self._ax.set_ylabel("Impact")

        self._ax.set_title(textwrap.fill(title, width=80))
        self._ax.set_xlabel(feature_name)
        self._ax.grid()

        for handle in self._ax.legend().legend_handles:
            handle._sizes = [25]

        # Format the axes.
        y_axis_formatter = self._axis_formatter(self._y_formatter)
        if y_axis_formatter is not None:
            self._ax.yaxis.set_major_formatter(y_axis_formatter)
        if self._x_formatters is not None and feature in self._x_formatters:
            x_formatter = self._x_formatters[feature]
        else:
            x_formatter = self._x_formatter_default
        x_axis_formatter = self._axis_formatter(x_formatter)
        if x_axis_formatter is not None:
            self._ax.xaxis.set_major_formatter(x_axis_formatter)

        if plot_id is not None:
            self._ax.text(
                0.99,
                0.02,
                plot_id,
                fontsize=8,
                backgroundcolor="white",
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=self._ax.transAxes,
            )

        return self._fig, self._ax
