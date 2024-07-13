from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Backend(ABC):

    @abstractmethod
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
    ):
        raise NotImplementedError
