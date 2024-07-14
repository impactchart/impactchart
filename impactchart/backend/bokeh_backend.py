"""Backend that uses bokeh for interactive impact charts."""

from typing import Optional

import pandas as pd
import textwrap

from ..backend.base import Backend


__bokeh_imported = False

try:
    from bokeh.plotting import figure

    __bokeh_imported = True
except ImportError:
    raise ImportError(
        "In order to import bokeh_backend, bokeh must be installed. "
        "Try `pip install impactchart[bokeh]` to enable interactive plots with bokeh."
    )


class BokehBackend(Backend):
    """A backend that plots interactive impact charts with bokeh."""

    def __init__(self):
        self._bokeh_plot = None
        self._ensemble_impact_label = None

    def begin(self):
        self._bokeh_plot = figure()

        self._ensemble_impact_label = "Impact of Individual Models"

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
        if y_name is not None:
            if subtitle is not None:
                title = f"Impact of {feature_name} on {y_name}\n{subtitle}"
            else:
                title = f"Impact of {feature_name} on {y_name}"
            self._bokeh_plot.yaxis.axis_label = f"Impact on {y_name}"
        else:
            if subtitle is not None:
                title = f"Impact of {feature_name} {subtitle}"
            else:
                title = f"Impact of {feature_name}"
            self._bokeh_plot.yaxis.axis_label = "Impact"

        self._bokeh_plot.title.text = textwrap.fill(title, width=80)
        self._bokeh_plot.xaxis.axis_label = feature_name

        return self._bokeh_plot

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
        if self._ensemble_impact_label is not None:
            self._bokeh_plot.scatter(
                x=x_feature,
                y=impact,
                color=ensemble_color,
                size=ensemble_marker_size,
                legend_label=self._ensemble_impact_label,
            )
        else:
            self._bokeh_plot.scatter(
                x=x_feature,
                y=impact,
                color=ensemble_color,
                size=ensemble_marker_size,
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
        self._bokeh_plot.scatter(
            x=x_feature,
            y=mean_impact,
            color=color,
            size=marker_size,
            legend_label="Median Impact",
        )
