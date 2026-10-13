from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


class DCCHEAVYRegressor:
    def __init__(
        self,
        return_col=0,
        rv_col=2,
        maxiter=1000,
        tol=1e-8,
        random_state=42,
    ):
        self.return_col = int(return_col)
        self.rv_col = int(rv_col)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.random_state = int(random_state)

    @staticmethod
    def _filter(r, x, theta):
        omega, alpha, gamma, beta = theta
        n = len(r)
        h = np.empty(n, dtype=float)
        h[0] = max(float(np.mean(x)), _EPS)
        for t in range(1, n):
            h[t] = max(
                omega + alpha * r[t - 1] ** 2 + gamma * x[t - 1] + beta * h[t - 1],
                _EPS,
            )
        return h

    @classmethod
    def _marginal_nll(cls, theta, r, x):
        omega, alpha, gamma, beta = theta
        if omega <= 0 or min(alpha, gamma, beta) < 0 or alpha + gamma + beta >= 0.999:
            return 1e50
        h = cls._filter(r, x, theta)
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + r**2 / h))

    @staticmethod
    def _dcc_nll(theta, z):
        a, b = theta
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e50

        z = np.asarray(z, dtype=float)
        s = np.corrcoef(z, rowvar=False)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(s, 1.0)
        q = s.copy()
        total = 0.0

        for t in range(1, z.shape[0]):
            outer = np.outer(z[t - 1], z[t - 1])
            q = (1.0 - a - b) * s + a * outer + b * q
            d = np.sqrt(np.maximum(np.diag(q), 1e-10))
            rmat = q / np.outer(d, d)
            rmat = (rmat + rmat.T) / 2.0
            rmat += np.eye(rmat.shape[0]) * 1e-8
            sign, logdet = np.linalg.slogdet(rmat)
            if sign <= 0:
                return 1e50
            try:
                inv = np.linalg.inv(rmat)
            except np.linalg.LinAlgError:
                return 1e50
            total += 0.5 * (logdet + float(z[t] @ inv @ z[t]))

        return float(total)

    @staticmethod
    def _panel_from_task_vectors(vectors, task_ids):
        lengths = {int(t): len(vectors[t]) for t in task_ids}
        min_len = min(lengths.values())
        if min_len < 30:
            raise ValueError(
                "DCC-HEAVY requires at least 30 synchronized observations per task."
            )

        panel = np.column_stack(
            [np.asarray(vectors[t], dtype=float)[-min_len:] for t in task_ids]
        )
        return panel, lengths, min_len

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        self.task_ids_ = [int(v) for v in np.unique(task)]
        self.params_ = {}
        self.next_h_ = {}
        residuals = {}

        for task_id in self.task_ids_:
            mask = task == task_id
            r = np.asarray(X[mask, self.return_col], dtype=float)
            rv = np.maximum(np.asarray(X[mask, self.rv_col], dtype=float), 1e-8)
            x = np.maximum(rv**2, _EPS)
            if len(r) < 30:
                raise ValueError(
                    f"DCC-HEAVY requires at least 30 observations for task {task_id}."
                )

            mean_x = max(float(np.mean(x)), 1e-10)
            x0 = np.array([0.03 * mean_x, 0.05, 0.10, 0.80], dtype=float)
            bounds = [
                (1e-12, max(10.0 * mean_x, 1e-6)),
                (1e-8, 0.95),
                (1e-8, 0.95),
                (1e-8, 0.98),
            ]
            constraints = (
                {"type": "ineq", "fun": lambda z: 0.999 - z[1] - z[2] - z[3]},
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step, clipping to bounds",
                    category=RuntimeWarning,
                )
                res = minimize(
                    self._marginal_nll,
                    x0,
                    args=(r, x),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": self.maxiter, "ftol": self.tol},
                )
            theta = np.asarray(res.x if res.success else x0, dtype=float)
            h = self._filter(r, x, theta)
            next_h = max(
                theta[0] + theta[1] * r[-1] ** 2 + theta[2] * x[-1] + theta[3] * h[-1],
                _EPS,
            )
            self.params_[task_id] = {
                "omega": float(theta[0]),
                "alpha": float(theta[1]),
                "gamma": float(theta[2]),
                "beta": float(theta[3]),
            }
            self.next_h_[task_id] = float(next_h)
            residuals[task_id] = r / np.sqrt(np.maximum(h, _EPS))

        z, self.panel_lengths_, self.dcc_panel_length_ = self._panel_from_task_vectors(
            residuals,
            self.task_ids_,
        )
        self.dcc_panel_was_balanced_ = len(set(self.panel_lengths_.values())) == 1

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step, clipping to bounds",
                category=RuntimeWarning,
            )
            dcc_res = minimize(
                self._dcc_nll,
                np.array([0.03, 0.94], dtype=float),
                args=(z,),
                method="SLSQP",
                bounds=[(1e-6, 0.98), (1e-6, 0.98)],
                constraints=({"type": "ineq", "fun": lambda q: 0.999 - q[0] - q[1]},),
                options={"maxiter": self.maxiter, "ftol": self.tol},
            )
        dcc = np.asarray(dcc_res.x if dcc_res.success else [0.03, 0.94], dtype=float)
        self.dcc_params_ = {"a": float(dcc[0]), "b": float(dcc[1])}
        self.dcc_converged_ = bool(dcc_res.success)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        pred = np.empty(len(X), dtype=float)

        for task_id in np.unique(task):
            key = int(task_id)
            idx = np.flatnonzero(task == task_id)
            p = self.params_[key]
            h_current = float(self.next_h_[key])

            for row_idx in idx:
                r_t = float(X[row_idx, self.return_col])
                rv_t = max(float(X[row_idx, self.rv_col]), 1e-8)
                x_t = rv_t**2
                h_next = max(
                    p["omega"]
                    + p["alpha"] * r_t**2
                    + p["gamma"] * x_t
                    + p["beta"] * h_current,
                    _EPS,
                )
                pred[row_idx] = np.sqrt(h_next)
                h_current = h_next

        return pred
