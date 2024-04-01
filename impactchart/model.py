"""
This package implements impact charts.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap.maskers
from matplotlib.ticker import Formatter, FuncFormatter, PercentFormatter
from scipy import stats
from shap import Explainer, TreeExplainer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
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
    optimize_hyperparameters
        Whether to optimize hyperparameters when fitting or just use the
        estimator with the given `estimator_kwargs`.
    """

    def __init__(
        self,
        *,
        ensemble_size: int = 50,
        training_fraction: float = 0.8,
        random_state: Optional[np.random.RandomState | int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        optimize_hyperparameters: bool = True,
    ):
        self._ensemble_size = ensemble_size
        self._training_fraction = training_fraction
        self._initial_random_state = random_state
        self._random_generator = np.random.default_rng(random_state)
        self._optimize_hyperparameters = optimize_hyperparameters

        self._plot_id = True

        if estimator_kwargs is None:
            estimator_kwargs = {}
        self._estimator_kwargs = estimator_kwargs

        self._ensembled_estimators = None

        self._X_fit = None

        # We will cache the impact here since it can be expensive
        # to compute. It's an LRU cache of size 1, which is easy
        # to implement and covers common cases. functools.lru_cache
        # does not work because the type pd.DataFrame is not hashable.
        self._df_impact = None
        self._X_for_last_df_impact = None

        # For reference, the r^2 score of the best model on the
        # full data set.
        self.r2_ = None
        self.best_score_ = None

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

    def ensemble_estimators(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        *,
        optimization_scoring_metric: Optional[str] = None,
    ) -> List[BaseEstimator]:
        """
        Construct all the estimators in the ensemble.

        Returns
        -------
            A list of estimators.
        """
        if self._optimize_hyperparameters:
            params = self.optimize_hyperparameters(
                X,
                y,
                sample_weight=sample_weight,
                optimization_scoring_metric=optimization_scoring_metric,
            )
            # Put the estimator kwargs back in. Mainly for things like
            # enable_categorical=True.
            params |= self._estimator_kwargs
        else:
            params = self._estimator_kwargs

        estimators = [self.estimator(**params) for _ in range(self._ensemble_size)]
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

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        *,
        optimization_scoring_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimize the hyperparameters of the base estimator.

        The default implementation does nothing. Derived classes can
        override this if they wish to optimize the hyperparameters
        of the estimator in an appropriate manner.

        Parameters
        ----------
        X
           Training features
        y
            Training values. Should have the same index as `X`
        sample_weight
            Sample weights. If provided, should have the same index as the `X`,
        optimization_scoring_metric
            The scoring metric to use. See https://scikit-learn.org/stable/modules/model_evaluation.html.
        Returns
        -------
            An optimized estimator. How, or even if, optimization is done is up to
            the derived concrete class.
        """
        return self._estimator_kwargs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        *,
        optimization_scoring_metric: Optional[str] = None,
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
        optimization_scoring_metric
            The scoring metric to use. See https://scikit-learn.org/stable/modules/model_evaluation.html.
        """
        self._ensembled_estimators = self.ensemble_estimators(
            X, y, sample_weight, optimization_scoring_metric=optimization_scoring_metric
        )

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

    @staticmethod
    def _estimator_predict(X: pd.DataFrame, estimator, estimator_id) -> pd.DataFrame:
        """
        Helper function for a single estimator.
        """
        df = pd.DataFrame(estimator.predict(X), columns=["y_hat"])
        df["estimator"] = estimator_id
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

    def mean_predicttion(self, X: pd.DataFrame) -> pd.Series:
        df_y_hat = self.predict(X)

        mean_y_hat = df_y_hat.groupby("X_index")["y_hat"].mean()

        mean_y_hat.index.name = X.index.name

        return mean_y_hat

    @staticmethod
    def _masker(X: pd.DataFrame) -> shap.maskers.Masker:
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

    def explainer(self, estimator, X: pd.DataFrame):
        return Explainer(
            estimator,
            masker=self._masker(X),
            algorithm=self.explainer_algorithm,
        )

    def _estimator_impact(
        self, X: pd.DataFrame, estimator, estimator_id
    ) -> pd.DataFrame:
        """
        Generate the impact of each feature for a given estimator.

        Parameters
        ----------
        X
            The feature values
        estimator
            The estimator
        estimator_id
            An id to put into the result so we can identify the estimator
            that the impacts came from.
        Returns
        -------
            A dataframe with columns for the impact of each feature and an added `id`
            column with the value we were given, for identification purposes.
        """
        df = pd.DataFrame(
            self.explainer(estimator, X)(X).values,
            columns=X.columns,
        )
        df["estimator"] = estimator_id
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
        if self._df_impact is None or not X.equals(self._X_for_last_df_impact):
            df_impact = pd.concat(
                self._estimator_impact(X, estimator, ii)
                for ii, estimator in enumerate(self._ensembled_estimators)
            )
            df_impact = df_impact.reset_index(names="X_index")

            self._df_impact = df_impact[["estimator", "X_index"] + list(X.columns)]
            self._X_for_last_df_impact = X

        return self._df_impact

    def mean_impact(self, X: pd.DataFrame) -> pd.DataFrame:
        df_impact = self.impact(X)

        df_mean_impact = df_impact.groupby("X_index")[list(X.columns)].mean()
        df_mean_impact.index.name = None

        return df_mean_impact

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

    def y_prime(
        self,
        X: pd.DataFrame,
        z_cols: Iterable[str],
        *,
        y: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute the repaired y values we should use for fitting a
        restorative model.

        Parameters
        ----------
        X
            The features to use for fitting the restorative model.
        z_cols
            The names of the protected features that we want to
            have no impact on the predictions of the revised model.
        y
            If given, these are the y values observed along with the
            X values. If not, then we will compute the y's based on
            what the fitted ensemble of models predicts (i.e. the
            return value of `self.predict(X)`. This is the base value
            that we will start with before restoring the impact of
            the protected features in `zcols`.
        Returns
        -------
        y_prime
            The y values we should use to fit a restorative model
            that so that protected features have little or no impact.
        """
        z_cols = set(z_cols)

        if y is not None:
            y_prime = y.copy()
        else:
            y_prime = self.mean_predicttion(X)

        y_prime.name = "y_prime"

        df_mean_impact = self.mean_impact(X)

        for col in df_mean_impact.columns:
            if col in z_cols:
                y_prime = y_prime - df_mean_impact[col]

        return y_prime

    def _plot_id(self, feature: str, n: int):
        if isinstance(self._initial_random_state, int):
            s = f"{self._initial_random_state:08X}"
        elif self._initial_random_state is None:
            s = "None"
        else:
            s = "???"

        return f"(f = {feature}; n = {n:,.0f}; k = {self.k}; s = {s})"

    def _plot_id_string(self, feature: str, n: int) -> str:
        msg = f"f = {feature}; n = {n:,.0f}; k = {self.k}; "

        if self._initial_random_state is not None:
            msg += f"s = {self._initial_random_state:08X}"
        else:
            msg += "s = None"

        return f"({msg})"

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

    def impact_charts(
        self,
        X: pd.DataFrame,
        features: Optional[Iterable[str]] = None,
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
        y_formatter: Optional[str] = None,
        x_formatter_default: str = "comma",
        x_formatters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Generate impact charts for a set of features.

        Parameters
        ----------
        X
            The feature values.
        features
            Which specific features we want charts for. If `None`, all columns of `X` will be assumed to be features.
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
        y_formatter
            How to format the y values. Can be one of the following: 'comma', 'percentage', 'dollar'.
        x_formatter_default:
            How to format the x axis unless uverridden for a particular feature by `x_formatters`.
            Same allowed values as `y_formatter`.
        x_formatters:
            A dictionary of how to format the x axis values. Keys are the features and values are the formats.
        Returns
        -------
            A dictionary whose key is the name of the features and whose values
            are tuples of `fiq` and `ax` for the plot for that feature.
        """
        if features is None:
            features = X.columns

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
            y_axis_formatter = self._axis_formatter(y_formatter)
            if y_axis_formatter is not None:
                ax.yaxis.set_major_formatter(y_axis_formatter)
            if x_formatters is not None and feature in x_formatters:
                x_formatter = x_formatters[feature]
            else:
                x_formatter = x_formatter_default
            x_axis_formatter = self._axis_formatter(x_formatter)
            if x_axis_formatter is not None:
                ax.xaxis.set_major_formatter(x_axis_formatter)

            if self._plot_id is not None:
                plot_id = self._plot_id_string(feature, len(X.index))
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

    def __init__(
        self,
        *,
        ensemble_size: int = 50,
        training_fraction: float = 0.8,
        random_state: Optional[np.random.RandomState | int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        optimize_hyperparameters: bool = True,
        parameter_distributions: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            training_fraction=training_fraction,
            random_state=random_state,
            estimator_kwargs=estimator_kwargs,
            optimize_hyperparameters=optimize_hyperparameters,
        )

        self._parameter_distributions = parameter_distributions

    def estimator(self, **kwargs) -> BaseEstimator:
        return XGBRegressor(**kwargs)

    def explainer(self, estimator, X: pd.DataFrame):
        has_categorical_features = False

        for col in X.columns:
            if X[col].dtype == "category":
                has_categorical_features = True
                break

        if has_categorical_features:
            # When categorical features are present, we have to uss
            # feature_perturbation="tree_path_dependent".
            return TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        else:
            # Otherwise, we will use the default feature_perturbation="interventional",
            # which will run slower but will do causal inference. A sample of 1,000
            # rows should be sufficient and will reduce large runtime overhead.
            if len(X.index) > 1000:
                X = X.sample(n=1000, random_state=self.initial_random_state)

            return TreeExplainer(estimator, X, feature_perturbation="interventional")

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        *,
        optimization_scoring_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimize the hyperparameters of the base estimator.

        Parameters
        ----------
        X
           Training features
        y
            Training values. Should have the same index as `X`
        sample_weight
            Sample weights. If provided, should have the same index as the `X`,
        optimization_scoring_metric
            The scoring metric to use. See https://scikit-learn.org/stable/modules/model_evaluation.html.
        Returns
        -------
            An optimized estimator.
        """
        param_dist = self._parameter_distributions

        if param_dist is None:
            param_dist = {
                "n_estimators": stats.randint(10, 100),
                "learning_rate": stats.uniform(0.01, 0.07),
                "subsample": stats.uniform(0.3, 0.7),
                "max_depth": stats.randint(2, 6),
                "min_child_weight": stats.randint(1, 4),
            }

        reg0 = XGBRegressor(**self._estimator_kwargs)

        reg = RandomizedSearchCV(
            reg0,
            param_distributions=param_dist,
            n_iter=200,
            error_score=0,
            n_jobs=-1,
            verbose=1,
            scoring=optimization_scoring_metric,
            random_state=int(self._random_generator.uniform(0.0, 1.0) * 4294967295),
        )

        reg.fit(X, y, sample_weight=sample_weight)

        self.r2_ = float(reg.best_estimator_.score(X, y, sample_weight=sample_weight))
        self.best_score_ = float(reg.best_score_)

        return reg.best_params_

    def _plot_id_string(self, feature: str, n: int) -> str:
        msg = f"f = {feature}; n = {n:,.0f}; k = {self.k}; "

        if self._initial_random_state is not None:
            msg += f"s = {self._initial_random_state:08X}"
        else:
            msg += "s = None"

        if self._optimize_hyperparameters:
            msg += f" | CV score = {self.best_score_:0.2f}; r2 = {self.r2_:0.2f})"
        return f"({msg})"


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
