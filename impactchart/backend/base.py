from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class Backend(ABC):

    def __init__(self):
        """
        An abstract base class for all plotting back ends.

        This class provides an interface that enables impact charts to be
        plotted on top of multiple static and interactive plotting tools
        such as matplotlib and bokeh.

        The interface consists of three methods. Additionaly, the constructor
        for a derived class will often have additional configuration arguments.

        The backend is used as follows:

        1. The caller constructs a backend with any configuration options
           they choose, depending on what the backend supports.
        2. The caller passes the backend they just constructed using the
           argument `backend=` to the method :py:meth:`~impactchart.ImpactModel.charts`.
        3. :py:meth:`~impactchart.ImpactModel.charts` calls the method
           :py:meth:`~impactchart.Backend.begin` to do any per-chart initialization.
        4. :py:meth:`~impactchart.ImpactModel.charts` calls the method
           :py:meth:`~impactchart.Backend.plot_ensemble_member_impact` to plot a series
           of 'k' impacts, one from each member of the ensemble of learners. This is done
           `n` times, once for each feature value in the input `X`.
        5. For each chart, For each chart, :py:meth:`~impactchart.ImpactModel.charts` calls the method
           :py:meth:`~impactchart.Backend.plot_mean_impact` to plot the mean impact
           series on top of the ensemble impacts.
        6. :py:meth:`~impactchart.ImpactModel.charts` calls the method
           :py:meth:`~impactchart.Backend.end`, which does any final work and returns
           a representation of the impact chart whose type is backend-dependend. The
           return value can be used to add further styling and display and/or save the chart.

        Note that steps 2-6 are repeated for each feature for which we want to generate
        an impact chart.
        """
        pass

    def begin(
        self,
        *,
        feature: str,
        feature_name: str,
    ):
        """
        Get ready to start plotting an import chart for a new feature.

        Parameters
        ----------
        feature
            The feature column.
        feature_name
            The name of the feature.

        Returns
        -------
            None

        """
        pass

    @abstractmethod
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
    ) -> Any:
        """
        Finalize a single impact chart and clean up any resources used to plot it.

        Parameters
        ----------
        feature
            The feature column.
        feature_name
            The name of the feature.
        min_impact
            The minimum value on the vertical axis of the impact chart.
        max_impact
            The maximum value on the vertical axis of the impact chart.
        y_name
            The name of the target for the impact chart.
        subtitle
            An optional subtitle for the impact chart
        plot_id
            An optional identifier to plot at the lower right of the impact chart.

        Returns
        -------
            A backend-specific representation of the impact chart that can be used to add further
            styling and display and/or save the chart.
        """
        raise NotImplementedError("Please use a concrete class.")

    @abstractmethod
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
        """
        Plot the impacts as computed by a single menber of the ensemble of `k` models.

        Parameters
        ----------
        x_feature
            The feature whose impact we are plotting.
        impact
            The impacts as computed by one member of the ensemble.
        index
            Which member of the enemble this was (out of `k`).
        feature_name
            The name of the feature.
        ensemble_marker_size
            The size of the marker to plot.
        ensemble_color
            The color of the marker to plot.

        Returns
        -------
            None
        """
        raise NotImplementedError("Please use a concrete class.")

    @abstractmethod
    def plot_mean_impact(
        self,
        x_feature: pd.Series,
        mean_impact: pd.Series,
        *,
        feature_name: Optional[str] = None,
        marker_size: float,
        color: str,
    ):
        """
        Plot the mean impact across all `k` members of the ensemble of models.

        Parameters
        ----------
        x_feature
            The feature whose impact we are plotting.
        mean_impact
            The mean impacts of each observation of the feature.
        feature_name
            The name of the feature.
        marker_size
            The size of the marker to plot.
        color
            The color of the marker to plot.

        Returns
        -------
            None
        """
        raise NotImplementedError("Please use a concrete class.")

    def plot(
        self,
        X: pd.DataFrame,
        df_impact: pd.DataFrame,
        feature: str,
        feature_name: str,
        *,
        min_impact: float,
        max_impact: float,
        marker_size: float = 4.0,
        ensemble_marker_size: float = 2.0,
        color: str,
        ensemble_color: str,
        y_name: Optional[str] = None,
        subtitle: Optional[str] = None,
        plot_id: Optional[str] = None,
    ):
        """
        Generate a plot by calling the various methods of a concrete class to do the backend-specific work.

        Parameters
        ----------
        X
            The features
        df_impact
            The impacts for all members of the ensemble of models
        feature
            The feature column.
        feature_name
            The name of the feature.
        min_impact
            The minimum value on the vertical axis of the impact chart.
        max_impact
            The maximum value on the vertical axis of the impact chart.
        marker_size
            The size of markers for mean impacts,
        ensemble_marker_size
            The size of markers for impacts for individual members of the ensemble of models,
        color
            The color of markers for mean impacts,
        ensemble_color
            The color of markers for impacts for individual members of the ensemble of models,
        y_name
            The name of the target for the impact chart.
        subtitle
            An optional subtitle for the impact chart
        plot_id
            An optional identifier to plot at the lower right of the impact chart.

        Returns
        -------
            A backend-specific representation of the impact chart that can be used to add further
            styling and display and/or save the chart.
        """
        self.begin(feature=feature, feature_name=feature_name)

        def _plot_for_ensemble_member(df_impact: pd.DataFrame):
            self.plot_ensemble_member_impact(
                X[feature],
                df_impact[feature],
                df_impact.name,
                feature_name=feature_name,
                ensemble_color=ensemble_color,
                ensemble_marker_size=ensemble_marker_size,
            )

        df_impact.groupby("estimator")[["X_index", feature]].apply(
            _plot_for_ensemble_member
        )

        mean_impact = df_impact.groupby("X_index")[feature].mean()

        self.plot_mean_impact(
            x_feature=X[feature],
            mean_impact=mean_impact,
            feature_name=feature_name,
            marker_size=marker_size,
            color=color,
        )

        return self.end(
            feature=feature,
            feature_name=feature_name,
            min_impact=min_impact,
            max_impact=max_impact,
            y_name=y_name,
            subtitle=subtitle,
            plot_id=plot_id,
        )
