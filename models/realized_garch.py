from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


class RealizedGARCHRegressor:
    def __init__(
        self,
        return_col=0,
        rv_col=2,
        maxiter=1500,
        tol=1e-8,
        random_state=42,
    ):
        self.return_col = int(return_col)
        self.rv_col = int(rv_col)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.random_state = int(random_state)

    @staticmethod
    def _unpack(theta):
        omega, beta, gamma, xi, phi, tau1, tau2, log_sigma = theta
        sigma_u = np.exp(log_sigma)
        return omega, beta, gamma, xi, phi, tau1, tau2, sigma_u

    @classmethod
    def _filter_logh(cls, theta, logx):
        omega, beta, gamma, *_ = cls._unpack(theta)
        n = len(logx)
        logh = np.empty(n, dtype=float)
        logh[0] = float(np.mean(logx))
        for t in range(1, n):
            logh[t] = omega + beta * logh[t - 1] + gamma * logx[t - 1]
            logh[t] = float(np.clip(logh[t], -30.0, 10.0))
        return logh

    @classmethod
    def _nll(cls, theta, r, logx):
        omega, beta, gamma, xi, phi, tau1, tau2, sigma_u = cls._unpack(theta)
        persistence = beta + gamma * phi
        if beta < 0 or gamma < 0 or phi <= 0 or sigma_u <= 0 or persistence >= 0.999:
            return 1e50

        logh = cls._filter_logh(theta, logx)
        h = np.exp(logh)
        z = r / np.sqrt(np.maximum(h, _EPS))
        measurement_mean = xi + phi * logh + tau1 * z + tau2 * (z**2 - 1.0)
        u = logx - measurement_mean

        return float(
            0.5 * np.sum(np.log(2.0 * np.pi) + logh + (r**2) / np.maximum(h, _EPS))
            + 0.5
            * np.sum(np.log(2.0 * np.pi) + 2.0 * np.log(sigma_u) + (u / sigma_u) ** 2)
        )

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        self.params_ = {}
        self.next_logh_ = {}
        self.converged_ = {}

        for task_id in np.unique(task):
            mask = task == task_id
            r = np.asarray(X[mask, self.return_col], dtype=float)
            rv = np.maximum(np.asarray(X[mask, self.rv_col], dtype=float), 1e-8)
            good = np.isfinite(r) & np.isfinite(rv)
            r = r[good]
            rv = rv[good]
            if len(r) < 30:
                raise ValueError(
                    f"Realized GARCH requires at least 30 observations for task {task_id}."
                )

            x = np.maximum(rv**2, _EPS)
            logx = np.log(x)
            mean_logx = float(np.mean(logx))
            x0 = np.array(
                [
                    0.05 * mean_logx,
                    0.65,
                    0.15,
                    0.0,
                    0.80,
                    0.0,
                    0.0,
                    np.log(max(float(np.std(logx, ddof=1)) * 0.25, 0.05)),
                ],
                dtype=float,
            )

            bounds = [
                (-10.0, 5.0),
                (0.0, 0.995),
                (0.0, 0.995),
                (-10.0, 10.0),
                (0.05, 3.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-8.0, 2.0),
            ]
            constraints = (
                {
                    "type": "ineq",
                    "fun": lambda z: 0.999 - (z[1] + z[2] * z[4]),
                },
            )

            res = minimize(
                self._nll,
                x0,
                args=(r, logx),
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

            logh = self._filter_logh(theta, logx)
            omega, beta, gamma, xi, phi, tau1, tau2, sigma_u = self._unpack(theta)
            next_logh = omega + beta * logh[-1] + gamma * logx[-1]
            next_logh = float(np.clip(next_logh, -30.0, 10.0))

            self.params_[int(task_id)] = {
                "omega": float(omega),
                "beta": float(beta),
                "gamma": float(gamma),
                "xi": float(xi),
                "phi": float(phi),
                "tau1": float(tau1),
                "tau2": float(tau2),
                "sigma_u": float(sigma_u),
            }
            self.next_logh_[int(task_id)] = next_logh
            self.converged_[int(task_id)] = bool(converged)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        pred = np.empty(len(X), dtype=float)

        for task_id in np.unique(task):
            key = int(task_id)
            idx = np.flatnonzero(task == task_id)
            p = self.params_[key]
            logh_current = float(self.next_logh_[key])

            for row_idx in idx:
                rv_t = max(float(X[row_idx, self.rv_col]), 1e-8)
                logx_t = np.log(max(rv_t**2, _EPS))
                logh_next = p["omega"] + p["beta"] * logh_current + p["gamma"] * logx_t
                logh_next = float(np.clip(logh_next, -30.0, 10.0))
                pred[row_idx] = np.sqrt(np.exp(logh_next))
                logh_current = logh_next

        return pred
