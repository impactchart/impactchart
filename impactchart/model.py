
from typing import Any, Dict, List, Optional, Tuple

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
import pandas as pd
from shap import Explainer, LinearExplainer
import shap.maskers


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
        self._random_state = random_state

        if estimator_kwargs is None:
            estimator_kwargs = {}
        self._estimator_kwargs = estimator_kwargs

        self._ensembled_estimators = self.ensemble_estimators()

        self._is_fit = False

    @abstractmethod
    def estimator(self, **kwargs) -> BaseEstimator:
        raise NotImplementedError("Abstract method.")

    def ensemble_estimators(self) -> List[BaseEstimator]:
        estimators = [self.estimator(**self._estimator_kwargs) for _ in range(self._ensemble_size)]
        return estimators

    def _training_sample(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        X_sample = X.sample(frac=self._training_fraction, random_state=self._random_state)
        # Once we have bootstrapped with a random state, we want to
        # let the random state evolve on its own.
        self._random_state = None
        y_sample = y.loc[X_sample.index]
        if sample_weight is not None:
            sample_weight_sample = sample_weight.loc[X_sample.index]
        else:
            sample_weight_sample = None

        return X_sample, y_sample, sample_weight_sample

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        for estimator in self._ensembled_estimators:
            X_sample, y_sample, sample_weight_sample = self._training_sample(X, y, sample_weight)
            if sample_weight is None:
                # Some estimators might not support sample weight
                estimator.fit(X_sample, y_sample)
            else:
                estimator.fit(X_sample, y_sample, sample_weight=sample_weight_sample)

        self._is_fit = True

    def _estimator_predict(self, X: pd.DataFrame, estimator, id) -> pd.DataFrame:
        df = pd.DataFrame(estimator.predict(X), columns=['y_hat'])
        df['estimator'] = id
        return df

    def predict(self, X: pd.DataFrame):
        df_y_hat = pd.concat(
            self._estimator_predict(X, estimator, ii)
            for ii, estimator in enumerate(self._ensembled_estimators)
        )

        df_y_hat = df_y_hat.reset_index(names="X_index")
        df_y_hat = df_y_hat[['estimator', 'X_index', 'y_hat']]

        return df_y_hat

    def _estimator_impact(self, X: pd.DataFrame, estimator, id) -> pd.DataFrame:
        df = pd.DataFrame(
            Explainer(estimator, shap.maskers.Independent(X, max_samples=1000))(X).values,
            columns=X.columns
        )
        df['estimator'] = id
        return df

    def impact(self, X: pd.DataFrame):
        df_impact = pd.concat(
            self._estimator_impact(X, estimator, ii)
            for ii, estimator in enumerate(self._ensembled_estimators)
        )

        df_impact = df_impact.reset_index(names="X_index")
        df_impact = df_impact[['estimator', 'X_index'] + list(X.columns)]

        return df_impact

    @property
    def coefs(self) -> List[np.array]:
        return [estimator.coef_ for estimator in self._ensembled_estimators]

    @property
    def intercepts(self) -> List:
        return [estimator.intercept_ for estimator in self._ensembled_estimators]

    @property
    def is_fit(self) -> bool:
        return self._is_fit


class LinearImpactModel(ImpactModel):

    def estimator(self, **kwargs) -> BaseEstimator:
        return LinearRegression(**kwargs)
