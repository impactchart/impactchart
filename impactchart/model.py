from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap.maskers
from shap import Explainer, KernelExplainer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


class ImpactModel(ABC):
    def __init__(
        self,
        *,
        ensemble_size: int = 50,
        training_fraction: float = 0.8,
        random_state: pd.core.common.RandomState = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._ensemble_size = ensemble_size
        self._training_fraction = training_fraction
        self._random_generator = np.random.default_rng(random_state)

        if estimator_kwargs is None:
            estimator_kwargs = {}
        self._estimator_kwargs = estimator_kwargs

        self._ensembled_estimators = self.ensemble_estimators()

        self._X_fit = None

    @abstractmethod
    def estimator(self, **kwargs) -> BaseEstimator:
        raise NotImplementedError("Abstract method.")

    def ensemble_estimators(self) -> List[BaseEstimator]:
        estimators = [
            self.estimator(**self._estimator_kwargs) for _ in range(self._ensemble_size)
        ]
        return estimators

    def _training_sample(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        X_sample = X.sample(
            frac=self._training_fraction, random_state=self._random_generator
        )

        y_sample = y.loc[X_sample.index]
        if sample_weight is not None:
            sample_weight_sample = sample_weight.loc[X_sample.index]
        else:
            sample_weight_sample = None

        return X_sample, y_sample, sample_weight_sample

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ):
        for estimator in self._ensembled_estimators:
            X_sample, y_sample, sample_weight_sample = self._training_sample(
                X, y, sample_weight
            )
            if sample_weight is None:
                # Some estimators might not support sample weight
                estimator.fit(X_sample, y_sample)
            else:
                estimator.fit(X_sample, y_sample, sample_weight=sample_weight_sample)

        self._X_fit = X

    def _estimator_predict(self, X: pd.DataFrame, estimator, id) -> pd.DataFrame:
        df = pd.DataFrame(estimator.predict(X), columns=["y_hat"])
        df["estimator"] = id
        return df

    def predict(self, X: pd.DataFrame):
        df_y_hat = pd.concat(
            self._estimator_predict(X, estimator, ii)
            for ii, estimator in enumerate(self._ensembled_estimators)
        )

        df_y_hat = df_y_hat.reset_index(names="X_index")
        df_y_hat = df_y_hat[["estimator", "X_index", "y_hat"]]

        return df_y_hat

    def masker(self, X: pd.DataFrame) -> shap.maskers.Masker:
        return shap.maskers.Independent(X, max_samples=1000)

    @property
    def explainer_algorithm(self) -> str:
        return "auto"

    def _estimator_impact(self, X: pd.DataFrame, estimator, id) -> pd.DataFrame:
        df = pd.DataFrame(
            Explainer(
                estimator,
                masker=self.masker(X),
                algorithm=self.explainer_algorithm,
            )(X).values,
            columns=X.columns,
        )
        df["estimator"] = id
        return df

    def impact(self, X: pd.DataFrame):
        df_impact = pd.concat(
            self._estimator_impact(X, estimator, ii)
            for ii, estimator in enumerate(self._ensembled_estimators)
        )

        df_impact = df_impact.reset_index(names="X_index")
        df_impact = df_impact[["estimator", "X_index"] + list(X.columns)]

        return df_impact

    def impact_charts(
        self,
        X: pd.DataFrame,
        features: Iterable[str],
        *,
        markersize: int = 4,
        color: str = "darkgreen",
        ensemble_markersize: int = 2,
        ensemble_color: str = "lightgray",
        plot_kwargs: Optional[Dict[str, Any]] = None,
        subplots_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        if plot_kwargs is None:
            plot_kwargs = {}

        if subplots_kwargs is None:
            subplots_kwargs = {}

        df_impact = self.impact(X)

        print()
        print(X)
        print()
        print(df_impact)
        print()

        impacts = {}

        features = list(features)

        for feature in features:
            fig, ax = plt.subplots(**subplots_kwargs)

            def _plot_for_ensemble_member(df_group):
                nonlocal plot_kwargs

                ax.plot(
                    X[feature],
                    df_group[feature],
                    ".",
                    markersize=ensemble_markersize,
                    color=ensemble_color,
                    **plot_kwargs,
                )
                plot_kwargs = {}

            df_impact.groupby("estimator")[["X_index", feature]].apply(
                _plot_for_ensemble_member
            )

            mean_impact = df_impact.groupby("X_index")[feature].mean()

            ax.plot(X[feature], mean_impact, ".", markersize=markersize, color=color)

            impacts[feature] = (
                fig,
                ax,
            )

        return impacts

    def impact_chart(
        self,
        X: pd.DataFrame,
        feature: str,
        *,
        markersize: int = 4,
        color: str = "darkgreen",
        ensemble_markersize: int = 2,
        ensemble_color: str = "lightgray",
        plot_kwargs: Optional[Dict[str, Any]] = None,
        subplots_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        return self.impact_charts(
            X,
            [feature],
            markersize=markersize,
            color=color,
            ensemble_markersize=ensemble_markersize,
            ensemble_color=ensemble_color,
            plot_kwargs=plot_kwargs,
            subplots_kwargs=subplots_kwargs,
        )[feature]

    @property
    def is_fit(self) -> bool:
        return self._X_fit is not None


class LinearImpactModel(ImpactModel):
    def estimator(self, **kwargs) -> BaseEstimator:
        return LinearRegression(**kwargs)

    @property
    def coefs(self) -> List[np.array]:
        return [estimator.coef_ for estimator in self._ensembled_estimators]

    @property
    def intercepts(self) -> List:
        return [estimator.intercept_ for estimator in self._ensembled_estimators]


class XGBoostImpactModel(ImpactModel):
    def estimator(self, **kwargs) -> BaseEstimator:
        return XGBRegressor(**kwargs)


class _CallableKNeighborsRegressor(KNeighborsRegressor):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class KnnImpactModel(ImpactModel):

    def estimator(self, **kwargs) -> BaseEstimator:
        estimator = _CallableKNeighborsRegressor(**kwargs)
        return estimator

    @property
    def explainer_algorithm(self) -> str:
        return "permutation"
