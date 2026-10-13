from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


class GARCHRegressor:

    def __init__(
        self,
        return_col=0,
        maxiter=1000,
        tol=1e-8,
        random_state=42,
    ):
        self.return_col = int(return_col)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.random_state = int(random_state)

    @staticmethod
    def _filter(r, omega, alpha, beta):
        r = np.asarray(r, dtype=float)
        n = len(r)
        h = np.empty(n, dtype=float)
        base = max(float(np.var(r, ddof=1)) if n > 1 else float(r[0] ** 2), _EPS)
        h[0] = base
        for t in range(1, n):
            h[t] = max(omega + alpha * r[t - 1] ** 2 + beta * h[t - 1], _EPS)
        return h

    @classmethod
    def _nll(cls, theta, r):
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e50
        h = cls._filter(r, omega, alpha, beta)
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + (r**2) / h))

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        self.params_ = {}
        self.next_h_ = {}
        self.converged_ = {}

        for task_id in np.unique(task):
            mask = task == task_id
            r = np.asarray(X[mask, self.return_col], dtype=float)
            r = r[np.isfinite(r)]
            if len(r) < 20:
                raise ValueError(
                    f"GARCH requires at least 20 observations for task {task_id}."
                )

            var = max(float(np.var(r, ddof=1)), 1e-10)
            x0 = np.array([0.05 * var, 0.08, 0.88], dtype=float)
            bounds = [(1e-12, max(10.0 * var, 1e-6)), (1e-8, 0.98), (1e-8, 0.98)]
            constraints = ({"type": "ineq", "fun": lambda z: 0.999 - z[1] - z[2]},)

            res = minimize(
                self._nll,
                x0,
                args=(r,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.maxiter, "ftol": self.tol},
            )

            if not res.success or not np.all(np.isfinite(res.x)):
                theta = x0
                converged = False
            else:
                theta = np.asarray(res.x, dtype=float)
                converged = True

            omega, alpha, beta = theta
            h = self._filter(r, omega, alpha, beta)
            next_h = max(omega + alpha * r[-1] ** 2 + beta * h[-1], _EPS)

            self.params_[int(task_id)] = {
                "omega": float(omega),
                "alpha": float(alpha),
                "beta": float(beta),
            }
            self.next_h_[int(task_id)] = float(next_h)
            self.converged_[int(task_id)] = bool(converged)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        pred = np.empty(len(X), dtype=float)

        for task_id in np.unique(task):
            key = int(task_id)
            if key not in self.params_:
                raise ValueError(f"Unknown task {key}.")

            idx = np.flatnonzero(task == task_id)
            p = self.params_[key]
            h_current = float(self.next_h_[key])

            for row_idx in idx:
                r_t = float(X[row_idx, self.return_col])
                h_next = max(
                    p["omega"] + p["alpha"] * r_t**2 + p["beta"] * h_current,
                    _EPS,
                )
                pred[row_idx] = np.sqrt(h_next)
                h_current = h_next

        return pred
