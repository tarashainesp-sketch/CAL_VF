from __future__ import annotations

import numpy as np
import pandas as pd


BO_MODEL_KEYS = {
    "gb",
    "rmtgb",
    "dnn_mtl",
    "gnn",
    "lstm",
}


def require_skopt():
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
        from skopt.utils import use_named_args
    except ImportError as exc:
        raise ImportError(
            "Install scikit-optimize: pip install scikit-optimize"
        ) from exc
    return gp_minimize, Integer, Real, Categorical, use_named_args


def default_space(model_key):
    _, Integer, Real, Categorical, _ = require_skopt()

    if model_key == "gb":
        return [
            Integer(100, 300, name="max_iter"),
            Real(0.01, 0.20, prior="log-uniform", name="learning_rate"),
            Real(0.0, 1.0, name="l2_regularization"),
        ]

    if model_key == "rmtgb":
        return [
            Categorical([40, 60, 80, 100, 120], name="n_iter_1st"),
            Categorical([20, 40, 60], name="n_iter_2nd"),
            Categorical([40, 60, 80, 100, 120], name="n_iter_3rd"),
            Real(0.01, 0.20, prior="log-uniform", name="learning_rate"),
        ]

    if model_key == "dnn_mtl":
        return [
            Categorical([0.001, 0.01, 0.1], name="learning_rate"),
        ]

    if model_key == "gnn":
        return [
            Categorical([16, 32, 64], name="hidden_dim"),
            Real(1e-4, 1e-2, prior="log-uniform", name="learning_rate"),
            Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
            Categorical([0.0, 0.15, 0.30, 0.45], name="graph_threshold"),
        ]

    if model_key == "lstm":
        return [
            Categorical([16, 32, 64], name="hidden_dim"),
            Categorical([4, 8, 12], name="lookback"),
            Real(1e-4, 1e-2, prior="log-uniform", name="learning_rate"),
            Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
        ]

    raise KeyError(f"No Bayesian-optimization space for {model_key!r}.")


def valid_configuration(model_key, params):
    if model_key == "rmtgb":
        total = (
            int(params["n_iter_1st"])
            + int(params["n_iter_2nd"])
            + int(params["n_iter_3rd"])
        )
        return 100 <= total <= 300
    return True


def bayes_search(
    model_key,
    factory,
    X_train,
    y_train,
    X_val,
    y_val,
    score_fn,
    n_calls,
    random_state,
):
    if model_key not in BO_MODEL_KEYS:
        raise KeyError(f"Bayesian optimization is not configured for {model_key!r}.")

    gp_minimize, _, _, _, use_named_args = require_skopt()
    space = default_space(model_key)
    history = []

    @use_named_args(space)
    def objective(**params):
        if not valid_configuration(model_key, params):
            history.append({**params, "validation_objective": 1e12, "valid": False})
            return 1e12
        try:
            model = factory(**params)
            model.fit(X_train.copy(), y_train)
            pred = np.asarray(model.predict(X_val.copy())).reshape(-1)
            score = float(score_fn(y_val, pred))
            if not np.isfinite(score):
                score = 1e12
            history.append({
                **params,
                "validation_objective": score,
                "valid": score < 1e12,
            })
            return score
        except Exception as exc:
            history.append({
                **params,
                "validation_objective": 1e12,
                "valid": False,
                "error": repr(exc),
            })
            return 1e12

    effective_calls = int(n_calls)
    if model_key == "dnn_mtl":
        effective_calls = max(3, min(effective_calls, 10))
    elif model_key in {"gnn", "lstm"}:
        effective_calls = max(8, min(effective_calls, 25))

    n_initial_points = min(
        effective_calls,
        min(10, max(3, effective_calls // 4)),
    )

    result = gp_minimize(
        objective,
        space,
        n_calls=effective_calls,
        n_initial_points=n_initial_points,
        random_state=int(random_state),
        acq_func="EI",
    )

    best = {
        dimension.name: value
        for dimension, value in zip(space, result.x)
    }
    history_df = pd.DataFrame(history)
    history_df.insert(0, "iteration", np.arange(1, len(history_df) + 1))
    return best, history_df
