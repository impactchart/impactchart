from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Backend(ABC):

    def begin(self):
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
    ):
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
        self.begin()

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
