"""Backend that uses bokeh for interactive impact charts."""
from typing import Optional

import pandas as pd

from ..backend.base import Backend


__bokeh_imported = False

try:
    from bokeh.plotting import figure, show

    __bokeh_imported = True
except ImportError:
    raise ImportError(
        "In order to import bokeh_backend, bokeh must be installed. "
        "If you pip installed impactchart, try `pip install impactchart[bokeh]` to enable interactive plots with bokeh."
    )


class BokehBackend(Backend):
    """A backend that plots interactive impact charts with bokeh."""

    def __init__(self):
        self._plot = None

    def begin(self):
        self._plot = figure()

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
        return self._plot

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
        self._plot.scatter(x=x_feature, y=impact)

    def plot_mean_impact(
            self,
            x_feature: pd.Series,
            mean_impact: pd.Series,
            *,
            feature_name: Optional[str] = None,
            marker_size: float,
            color: str,
    ):
        pass
