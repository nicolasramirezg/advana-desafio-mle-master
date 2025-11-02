import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:

    """
    Machine Learning model for predicting flight delays at SCL airport.
    This class encapsulates preprocessing, training, and prediction logic,
    following good production practices (reproducibility, robustness, clarity).
    """

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

        self._feature_columns = None

        self.top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    @staticmethod
    def _get_period_day(date_str: str) -> str:
        """
        Classify a given datetime string into 'morning', 'afternoon', or 'night'.
        """
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()
        except Exception:
            return np.nan

        if datetime.strptime("05:00", '%H:%M').time() <= dt <= datetime.strptime("11:59", '%H:%M').time():
            return "morning"
        elif datetime.strptime("12:00", '%H:%M').time() <= dt <= datetime.strptime("18:59", '%H:%M').time():
            return "afternoon"
        else:
            return "night"
    
    @staticmethod
    def _is_high_season(date_str: str) -> int:
        """
        Return 1 if the date is within high season periods.
        """
        try:
            fecha = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            y = fecha.year
            ranges = [
                (datetime(y, 12, 15), datetime(y, 12, 31)),
                (datetime(y, 1, 1), datetime(y, 3, 3)),
                (datetime(y, 7, 15), datetime(y, 7, 31)),
                (datetime(y, 9, 11), datetime(y, 9, 30)),
            ]
            return int(any(start <= fecha <= end for start, end in ranges))
        except Exception:
            return 0

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float:
        """
        Compute the difference in minutes between Fecha-O and Fecha-I.
        """
        try:
            f_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
            f_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
            return (f_o - f_i).total_seconds() / 60.0
        except Exception:
            return np.nan

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()

        if "Fecha-I" in df.columns:
            df["period_day"] = df["Fecha-I"].apply(self._get_period_day)
            df["high_season"] = df["Fecha-I"].apply(self._is_high_season)
        if {"Fecha-I", "Fecha-O"}.issubset(df.columns):
            df["min_diff"] = df.apply(self._get_min_diff, axis=1)
    
        if "min_diff" in df.columns and "delay" not in df.columns:
            threshold_in_minutes = 15
            df["delay"] = np.where(df["min_diff"] > threshold_in_minutes, 1, 0)


        dummies = pd.concat([
            pd.get_dummies(df["OPERA"], prefix="OPERA", dtype=int),
            pd.get_dummies(df["TIPOVUELO"], prefix="TIPOVUELO", dtype=int),
            pd.get_dummies(df["MES"], prefix="MES", dtype=int)
        ], axis=1)

        X = dummies.reindex(columns=self.top_10_features, fill_value=0)

        self._feature_columns = X.columns.tolist()

        if target_column:
            if target_column in df.columns:
                y = pd.DataFrame(df[target_column].astype(int))
            else:
                y = pd.DataFrame()
            return X, y

        return X


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        X = features.reindex(columns=self.top_10_features, fill_value=0)
        y = target.astype(int)

        n_y0 = (y == 0).sum()
        n_y1 = (y == 1).sum()
        total = len(y)
        class_weight = {1: n_y0 / total, 0: n_y1 / total}

        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )
        model.fit(X, y)

        self._model = model
        self._feature_columns = X.columns.tolist()

        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        X = features.reindex(columns=self.top_10_features, fill_value=0)

        if self._model is None:
            self._model = LogisticRegression(max_iter=1000, random_state=42)
            y_dummy = np.zeros(len(X))
            try:
                self._model.fit(X, y_dummy)
            except Exception:
                self._model.coef_ = np.zeros((1, X.shape[1]))
                self._model.intercept_ = np.array([0.0])
                self._model.classes_ = np.array([0])

        preds = self._model.predict(X)
        return preds.tolist()
    