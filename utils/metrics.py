from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def score_regression(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_by_task(model_name, model, X, y, task_names, split="test"):
    pred = np.asarray(model.predict(X.copy())).reshape(-1)
    task_ids = X[:, -1].astype(int)
    rows = [
        {
            "model": model_name,
            "split": split,
            "task": "ALL",
            "n": int(len(y)),
            **score_regression(y, pred),
        }
    ]
    task_scores = []
    for task_id, task_name in task_names.items():
        idx = task_ids == int(task_id)
        if not np.any(idx):
            continue
        m = score_regression(y[idx], pred[idx])
        task_scores.append(m)
        rows.append(
            {
                "model": model_name,
                "split": split,
                "task": task_name,
                "n": int(idx.sum()),
                **m,
            }
        )
    if task_scores:
        rows.append(
            {
                "model": model_name,
                "split": split,
                "task": "TASK_AVG",
                "n": int(len(y)),
                **{
                    k: float(np.mean([r[k] for r in task_scores]))
                    for k in ("RMSE", "MAE", "R2")
                },
            }
        )
    pdf = pd.DataFrame(
        {
            "task_id": task_ids,
            "task": [task_names.get(int(t), str(int(t))) for t in task_ids],
            "y_true": np.asarray(y).reshape(-1),
            "y_pred": pred,
        }
    )
    pdf["error"] = pdf["y_true"] - pdf["y_pred"]
    pdf["abs_error"] = pdf["error"].abs()
    pdf["squared_error"] = pdf["error"] ** 2
    return pd.DataFrame(rows), pdf
