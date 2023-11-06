"""
This package implements impact charts.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap.maskers
from shap import Explainer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


class ImpactModel(ABC):
    """
    An abstract base class for all impact models. This class contains
    the core functionality for producing impact charts.

    An impact chart is a chart that shows, for a given data set,
    how the predictions of a model are impacted by the values of a
    particular feature in that data set. See
    https://datapinions.com/using-interpretable-machine-learning-to-analyze-racial-and-ethnic-disparities-in-home-values/
    for an introduction to impact graphs and their application.

    An impact model is, at it's core, an ensemble of machine learning
    models. It is up to derived classes to decide what kind
    of models go into the ensemble. They could, for example, be
    linear models, tree-based models, or nearest neighbor models.
    Each model in the ensemble is trained on a different random subset
    of the overall training data available to the impact model. In general
    these subsets ovelap.

    The impact model uses Shapley values from each of the predictive models
    in the ensemble to produce impact charts.

    Parameters
    ----------
    ensemble_size
        How many models to ensemble together to generate impact charts.
    training_fraction
        The fraction of the data that should be used to train each model.
    random_state
        A random state. Using the same initial random state should cause
        reproducible deterministic outcomes.
    estimator_kwargs
        Keyword args to be passed to the underlying models.
    """

    def __init__(
        self,
        *,
        ensemble_size: int = 50,
        training_fraction: float = 0.8,
        random_state: Optional[np.random.RandomState | int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._ensemble_size = ensemble_size
        self._training_fraction = training_fraction
        self._initial_random_state = random_state
        self._random_generator = np.random.default_rng(random_state)

        if estimator_kwargs is None:
            estimator_kwargs = {}
        self._estimator_kwargs = estimator_kwargs

        self._ensembled_estimators = self.ensemble_estimators()

        self._X_fit = None

        # We will cache the impact here since it can be expensive
        # to compute.
        self._df_impact = None

    @property
    def k(self) -> int:
        return self._ensemble_size

    @property
    def initial_random_state(self) -> np.random.RandomState | None:
        return self._initial_random_state

    @abstractmethod
    def estimator(self, **kwargs) -> BaseEstimator:
        """
        Construct a single underlying estimator for the ensemble.
        Each derived class should implement this to construct an
        estimator of an appropriate class.

        Parameters
        ----------
        kwargs
            Keyword args to construct the underlying estimator.

        Returns
        -------
            An estimator.
        """
        raise NotImplementedError("Abstract method.")

    def ensemble_estimators(self) -> List[BaseEstimator]:
        """
        Construct all the estimators in the ensemble.

        Returns
        -------
            A list of estimators.
        """
        estimators = [
            self.estimator(**self._estimator_kwargs) for _ in range(self._ensemble_size)
        ]
        return estimators

    def _training_sample(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        """
        A helper function to create a sample to train one model in the ensemble.

        Parameters
        ----------
        X
            Training features
        y
            Corresponding values. Should have the same index as `X`.
        sample_weight
            Optional weights. If provided, should have the same index as `X`.

        Returns
        -------
            `X_sample`, `y_sample`, `sample_weight_sample`
            A tuple of samples of the three parameters passed in.
        """
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
        """
        Fit all of the estimators in the ensemble so that we can
        subsequently produce an impact chart.

        Parameters
        ----------
        X
            Training features.
        y
            Corresponding values. Should have the same index as `X`.
        sample_weight
            Optional weights. If provided, should have the same index as `X`.
        """
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
        """
        Helper function for a single estimator.
        """
        df = pd.DataFrame(estimator.predict(X), columns=["y_hat"])
        df["estimator"] = id
        return df

    def predict(self, X: pd.DataFrame):
        """
        Make predictions for every element of the ensemble.

        Parameters
        ----------
        X
            The input features

        Returns
        -------
            A data frame with predictions from each of the estimators.
            The columns in the returned value are `"estimator"`, which is
            a unique id for each estimator in the ensemble, `'X_index'`, which
            is the index value of the input `X`, and `'y_hat'` which is
            the corresponding prediction. If there are n rows in `X` and
            k estimators then the return value has n * k rows.
        """
        df_y_hat = pd.concat(
            self._estimator_predict(X, estimator, ii)
            for ii, estimator in enumerate(self._ensembled_estimators)
        )

        df_y_hat = df_y_hat.reset_index(names="X_index")
        df_y_hat = df_y_hat[["estimator", "X_index", "y_hat"]]

        return df_y_hat

    def _masker(self, X: pd.DataFrame) -> shap.maskers.Masker:
        """
        SHAP masker.

        It's possible we might promote this to a public method if it
        becomes important to control the masker.
        """
        return shap.maskers.Independent(X, max_samples=1000)

    @property
    def explainer_algorithm(self) -> str:
        """
        What SHAP explainer algorithm to use.

        Returns
        -------
            The name of the algorithm.
        """
        return "auto"

    def _estimator_impact(self, X: pd.DataFrame, estimator, id) -> pd.DataFrame:
        """
        Generate the impact of each feature for a given estimator.

        Parameters
        ----------
        X
            The feature values
        estimator
            The estimator
        id
            An id to put into the result so we can identify the estimator
            that the impacts came from.
        Returns
        -------
            A dataframe with columns for the impact of each feature and an added `id`
            column with the value we were given, for identification purposes.
        """
        df = pd.DataFrame(
            Explainer(
                estimator,
                masker=self._masker(X),
                algorithm=self.explainer_algorithm,
            )(X).values,
            columns=X.columns,
        )
        df["estimator"] = id
        return df

    def impact(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a data frame of impacts for each feature according to each estimator.

        Parameters
        ----------
        X
            The feature values.
        Returns
        -------
            A dataframe of impacts with one column for each feature impact and an id
            column. The number of rows in the number of rows in X times the number of
            estimators.
        """
        if self._df_impact is None:
            df_impact = pd.concat(
                self._estimator_impact(X, estimator, ii)
                for ii, estimator in enumerate(self._ensembled_estimators)
            )

            df_impact = df_impact.reset_index(names="X_index")
            self._df_impact = df_impact[["estimator", "X_index"] + list(X.columns)]

        return self._df_impact

    def bucketed_impact(
        self, X: pd.DataFrame, feature: str, buckets: int = 10
    ) -> pd.DataFrame:
        df_impact = self.impact(X)

        df_mean_impact = (
            df_impact.groupby("X_index")[[feature]]
            .mean()
            .rename({feature: "impact"}, axis="columns")
        )

        df_value_and_mean_impact = X[[feature]].join(df_mean_impact)

        df_value_and_mean_impact.sort_values(by=feature, inplace=True)

        n = len(df_value_and_mean_impact.index)

        df_value_and_mean_impact["bucket"] = [ii for ii in range(n)]

        df_value_and_mean_impact["bucket"] = (
            df_value_and_mean_impact["bucket"] // (n * 1.0 / buckets)
        ).astype(int)

        df_buckets = df_value_and_mean_impact.groupby("bucket").agg(
            {feature: "max", "impact": "mean"}
        )

        return df_buckets

    def _plot_id(self, feature: str, n: int):
        if isinstance(self._initial_random_state, int):
            s = f"{self._initial_random_state:08X}"
        elif self._initial_random_state is None:
            s = "None"
        else:
            s = "???"

        return f"(f = {feature}; n = {n:,.0f}; k = {self.k}; s = {s})"

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
        feature_names: Optional[Callable[[str], str] | Mapping[str, str]] = None,
        y_name: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Generate impact charts for a set of features.

        Parameters
        ----------
        X
            The feature values.
        features
            Which specific features we want charts for.
        markersize
            Size of the marker
        color
            Color of the marker
        ensemble_markersize
            Size of the marker for the mean of the ensemble
        ensemble_color
            Color of the marker for the mean of the ensemble
        plot_kwargs
            Additional keyword args for matplotlib
        subplots_kwargs
            Additional kwargs for `plt.subplots` call to create the subplots.
        feature_names
            A map of function from features to the names to use for them
            in titles in the chart.
        y_name
            A name to use for the output of the model.
        subtitle
            A subtitle for the plot.
        Returns
        -------
            A dictionary whose key is the name of the features and whose values
            are tuples of `fiq` and `ax` for the plot for that feature.
        """
        if plot_kwargs is None:
            plot_kwargs = {}

        if subplots_kwargs is None:
            subplots_kwargs = {}

        df_impact = self.impact(X)

        # We want to scale the y axis of all of the charts the same,
        # based on the global min and max impact, so we can easily
        # compare them visually.
        max_impact = df_impact[features].max(axis="columns").max(axis="rows")
        min_impact = df_impact[features].min(axis="columns").min(axis="rows")

        impact_span = max_impact - min_impact

        max_impact = max_impact + 0.05 * impact_span
        min_impact = min_impact - 0.05 * impact_span

        impacts = {}

        features = list(features)

        if feature_names is None:
            # We got not mapping or function, so just use the feature name.
            def feature_name_func(f):
                return f

        elif callable(feature_names):
            # We got a callable
            feature_name_func = feature_names
        else:
            # Expect it to be a map.
            def feature_name_func(f):
                return feature_names[f]

        for feature in features:
            # Only label the first series with ensemble
            # impact so the legend stays just two entries.
            ensemble_impact_label = "Impact of Individual Models"

            feature_name = feature_name_func(feature)

            fig, ax = plt.subplots(**subplots_kwargs)

            def _plot_for_ensemble_member(df_group):
                nonlocal plot_kwargs, ensemble_impact_label

                plot_x = X[feature]
                plot_y = df_group[feature]

                ax.plot(
                    plot_x,
                    plot_y,
                    ".",
                    markersize=ensemble_markersize,
                    color=ensemble_color,
                    label=ensemble_impact_label,
                    **plot_kwargs,
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
                markersize=markersize,
                color=color,
                label="Mean Impact",
            )

            # Do some basic labels and styling.
            ax.set_ylim(min_impact, max_impact)

            if y_name is not None:
                if subtitle is not None:
                    ax.set_title(f"Impact of {feature_name}\non {y_name}\n{subtitle}")
                else:
                    ax.set_title(f"Impact of {feature_name}\non {y_name}")
                ax.set_ylabel(f"Impact on {y_name}")
            else:
                if subtitle is not None:
                    ax.set_title(f"Impact of {feature_name}\n{subtitle}")
                else:
                    ax.set_title(f"Impact of {feature_name}")
                ax.set_ylabel("Impact")
            ax.set_xlabel(feature_name)
            ax.grid()

            for handle in ax.legend().legend_handles:
                handle._sizes = [25]

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
        """
        A single impact chart for a single feature.

        Parameters
        ----------
        X
            The feature values.
        feature
            The feature to get a plot for/
        markersize
            Size of the marker
        color
            Color of the marker
        ensemble_markersize
            Size of the marker for the mean of the ensemble
        ensemble_color
            Color of the marker for the mean of the ensemble
        plot_kwargs
            Additional keyword args for matplotlib
        subplots_kwargs
            Additional kwargs for `plt.subplots` call to create the subplots.

        Returns
        -------
            `fig`, `ax` for the plot.
        """
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
        """Have we been fit? `True` or `False`"""
        return self._X_fit is not None


class LinearImpactModel(ImpactModel):
    """
    An :py:class:`~ImpactModel` that uses linear estimators.
    """

    def estimator(self, **kwargs) -> BaseEstimator:
        return LinearRegression(**kwargs)

    @property
    def coefs(self) -> List[np.array]:
        return [estimator.coef_ for estimator in self._ensembled_estimators]

    @property
    def intercepts(self) -> List:
        return [estimator.intercept_ for estimator in self._ensembled_estimators]


class XGBoostImpactModel(ImpactModel):
    """
    An :py:class:`~ImpactModel` that uses XGBoost estimators.
    """

    def estimator(self, **kwargs) -> BaseEstimator:
        return XGBRegressor(**kwargs)


class _CallableKNeighborsRegressor(KNeighborsRegressor):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class KnnImpactModel(ImpactModel):
    """
    An :py:class:`~ImpactModel` that uses k nearest neighbor estimators.
    """

    def estimator(self, **kwargs) -> BaseEstimator:
        estimator = _CallableKNeighborsRegressor(**kwargs)
        return estimator

    @property
    def explainer_algorithm(self) -> str:
        return "permutation"
