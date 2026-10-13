from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


class WeeklyHARRVRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, rv_start_idx=2, fit_intercept=True):
        self.rv_start_idx = rv_start_idx
        self.fit_intercept = fit_intercept

    def _har_features(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)

        start = int(self.rv_start_idx)
        stop = start + 4
        if data.ndim != 2 or data.shape[1] < stop:
            raise ValueError(
                "WeeklyHARRVRegressor requires four consecutive RV columns "
                f"starting at rv_start_idx={start}. Received shape {data.shape}."
            )

        rv = data[:, start:stop]
        recent = rv[:, 0]
        short = rv[:, :2].mean(axis=1)
        medium = rv.mean(axis=1)

        return np.column_stack([recent, short, medium])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be a 2-D array with task id in the last column.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of observations.")

        task = X[:, -1].astype(int)
        data = X[:, :-1]
        har_x = self._har_features(data)

        self.models_ = {}
        self.task_values_ = np.sort(np.unique(task))

        for t in self.task_values_:
            mask = task == t
            model = LinearRegression(fit_intercept=bool(self.fit_intercept))
            model.fit(har_x[mask], y[mask])
            self.models_[int(t)] = model

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "models_")

        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        data = X[:, :-1]
        har_x = self._har_features(data)

        pred = np.empty(len(X), dtype=float)
        for t in np.unique(task):
            key = int(t)
            if key not in self.models_:
                raise ValueError(f"No HAR-RV model was fitted for task {key}.")
            mask = task == t
            pred[mask] = self.models_[key].predict(har_x[mask])

        return pred

    def get_task_coefficients(self):
        check_is_fitted(self, "models_")
        out = {}
        for task, model in self.models_.items():
            out[int(task)] = {
                "intercept": float(np.asarray(model.intercept_).reshape(-1)[0]),
                "recent": float(model.coef_[0]),
                "short": float(model.coef_[1]),
                "medium": float(model.coef_[2]),
            }
        return out
