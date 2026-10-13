from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import root_mean_squared_error
from .metrics import evaluate_by_task
from .plots import (
    save_metric_bars,
    save_prediction_scatter,
    save_error_distribution,
    save_theta_plot,
    save_rmb_dendrogram,
)


def rmse_objective(y_true, y_pred):
    return float(root_mean_squared_error(y_true, y_pred))


def _j(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def extract_artifacts(model, task_names):
    out = {}
    if hasattr(model, "theta"):
        th = np.asarray(model.theta, float).reshape(-1)
        sg = expit(th)
        out["theta"] = pd.DataFrame(
            [
                {
                    "task_id": i,
                    "task": task_names.get(i, str(i)),
                    "theta": float(v),
                    "sigmoid_theta": float(sg[i]),
                    "weight_nonout_common": float(1 - sg[i]),
                    "weight_out_common": float(sg[i]),
                }
                for i, v in enumerate(th)
            ]
        )
    if hasattr(model, "task_to_cluster_"):
        out["clusters"] = pd.DataFrame(
            [
                {
                    "task_id": int(k),
                    "task": task_names.get(int(k), str(int(k))),
                    "cluster": int(v),
                }
                for k, v in sorted(model.task_to_cluster_.items())
            ]
        )
        out["rmb_meta"] = {
            "task_to_cluster": {
                task_names.get(int(k), str(int(k))): int(v)
                for k, v in sorted(model.task_to_cluster_.items())
            },
            "distance_matrix": np.asarray(
                getattr(model, "distance_matrix_", []), float
            ).tolist(),
            "similarity_matrix": np.asarray(
                getattr(model, "residual_matrix_", []), float
            ).tolist(),
            "linkage_matrix": np.asarray(
                getattr(model, "linkage_matrix_", []), float
            ).tolist(),
        }
    if hasattr(model, "get_training_history"):
        out["history"] = model.get_training_history()
    return out


def train_one_model(
    model_key,
    display_name,
    factory,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    task_names,
    model_dir,
    tuning_metadata=None,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    model = factory(**params)
    model.fit(X_train.copy(), y_train)
    seconds = time.time() - t
    metrics, preds = evaluate_by_task(display_name, model, X_test, y_test, task_names)
    metrics.to_csv(model_dir / "metrics.csv", index=False)
    preds.to_csv(model_dir / "predictions.csv", index=False)
    a = extract_artifacts(model, task_names)
    if "theta" in a:
        a["theta"].to_csv(model_dir / "theta.csv", index=False)
    if "clusters" in a:
        a["clusters"].to_csv(model_dir / "clusters.csv", index=False)
    if "history" in a:
        pd.DataFrame(a["history"]).to_csv(
            model_dir / "training_history.csv", index=False
        )
    meta = {
        "model_key": model_key,
        "display_name": display_name,
        "fit_seconds": float(seconds),
        "parameters": {k: _j(v) for k, v in params.items()},
        "tuning": tuning_metadata,
    }
    if "rmb_meta" in a:
        meta["rmb_cle"] = a["rmb_meta"]
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_metric_bars(metrics, model_dir)
    save_prediction_scatter(preds, model_dir)
    save_error_distribution(preds, model_dir)
    if "theta" in a:
        save_theta_plot(a["theta"], model_dir)
    if model_key == "rmb_cle":
        save_rmb_dendrogram(model, task_names, model_dir)
    return model, metrics, preds, meta
