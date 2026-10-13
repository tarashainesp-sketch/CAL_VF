from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_task_name(task_id, tasks):
    return tasks.get(int(task_id), str(task_id))


def _cosine_similarity_matrix(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    z = x / norms
    return z @ z.T


def _to_1d_prediction(pred):
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred.reshape(-1)


def _save_rmtgb_gates(model, tasks, output_dir):
    output_dir = Path(output_dir)

    theta = None
    source_name = None

    for name in (
        "theta",
        "theta_",
        "thetas_",
        "task_theta_",
        "gating_weights_",
        "gate_weights_",
    ):
        if hasattr(model, name):
            theta = getattr(model, name)
            source_name = name
            break

    if theta is None:
        return None

    if isinstance(theta, dict):
        keys = list(theta.keys())
        values = np.asarray([theta[k] for k in keys], dtype=float).reshape(-1)
        task_ids = [int(k) for k in keys]
    else:
        values = np.asarray(theta, dtype=float).reshape(-1)

        if hasattr(model, "tasks_dic"):
            mapping = getattr(model, "tasks_dic")
            inverse = {int(v): int(k) for k, v in mapping.items()}
            task_ids = [inverse.get(i, i) for i in range(len(values))]
        else:
            task_ids = list(range(len(values)))

    rows = []
    for task_id, value in zip(task_ids, values):
        sigmoid_value = 1.0 / (1.0 + np.exp(-float(value)))
        rows.append(
            {
                "task_id": int(task_id),
                "task": _safe_task_name(task_id, tasks),
                "theta": float(value),
                "sigmoid_theta": float(sigmoid_value),
                "source_attribute": source_name,
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "rmtgb_gating_weights.csv"
    df.to_csv(path, index=False)
    return path


def save_rmtgb_diagnostics(model, X, y, tasks, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    task_id = X[:, -1].astype(int)

    n1 = int(getattr(model, "n_iter_1st"))
    n2 = int(getattr(model, "n_iter_2nd"))
    n3 = int(getattr(model, "n_iter_3rd"))

    if not hasattr(model, "_staged_raw_predict"):
        raise AttributeError(
            "R-MTGB does not expose _staged_raw_predict, so the three boosting "
            "blocks cannot be decomposed without modifying the original implementation."
        )

    staged = []
    for pred in model._staged_raw_predict(X.copy()):
        staged.append(_to_1d_prediction(pred).copy())

    required = n1 + n2 + n3
    if len(staged) < required:
        raise RuntimeError(
            f"R-MTGB produced {len(staged)} staged predictions, but "
            f"{required} were expected from n1+n2+n3."
        )

    pred_shared = staged[n1 - 1]
    pred_shared_plus_adaptive = staged[n1 + n2 - 1]
    pred_full = staged[n1 + n2 + n3 - 1]

    adaptive_component = pred_shared_plus_adaptive - pred_shared
    task_specific_component = pred_full - pred_shared_plus_adaptive

    component_df = pd.DataFrame(
        {
            "task_id": task_id,
            "task": [_safe_task_name(t, tasks) for t in task_id],
            "y_true": y,
            "shared_prediction": pred_shared,
            "adaptive_shared_component": adaptive_component,
            "task_specific_component": task_specific_component,
            "full_prediction": pred_full,
        }
    )
    component_df.to_csv(
        output_dir / "rmtgb_component_predictions.csv",
        index=False,
    )

    rows = []
    for tid in sorted(np.unique(task_id)):
        mask = task_id == tid
        yt = y[mask]

        shared = pred_shared[mask]
        adaptive = adaptive_component[mask]
        specific = task_specific_component[mask]
        full = pred_full[mask]

        mean_abs_shared = float(np.mean(np.abs(shared)))
        mean_abs_adaptive = float(np.mean(np.abs(adaptive)))
        mean_abs_specific = float(np.mean(np.abs(specific)))

        total_abs = mean_abs_shared + mean_abs_adaptive + mean_abs_specific
        if total_abs > 1e-12:
            shared_fraction = mean_abs_shared / total_abs
            adaptive_fraction = mean_abs_adaptive / total_abs
            specific_fraction = mean_abs_specific / total_abs
        else:
            shared_fraction = np.nan
            adaptive_fraction = np.nan
            specific_fraction = np.nan

        without_task_specific = shared + adaptive
        without_adaptive = shared + specific

        rmse_full = _rmse(yt, full)
        rmse_shared_only = _rmse(yt, shared)
        rmse_without_task = _rmse(yt, without_task_specific)
        rmse_without_adaptive = _rmse(yt, without_adaptive)

        rows.append(
            {
                "task_id": int(tid),
                "task": _safe_task_name(tid, tasks),
                "mean_abs_shared": mean_abs_shared,
                "mean_abs_adaptive_shared": mean_abs_adaptive,
                "mean_abs_task_specific": mean_abs_specific,
                "shared_fraction": shared_fraction,
                "adaptive_shared_fraction": adaptive_fraction,
                "task_specific_fraction": specific_fraction,
                "rmse_full": rmse_full,
                "rmse_shared_only": rmse_shared_only,
                "rmse_without_adaptive": rmse_without_adaptive,
                "rmse_without_task_specific": rmse_without_task,
                "adaptive_shared_rmse_gain": rmse_without_adaptive - rmse_full,
                "task_specific_rmse_gain": rmse_without_task - rmse_full,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_dir / "rmtgb_sharing_diagnostics.csv",
        index=False,
    )

    _save_rmtgb_gates(
        model=model,
        tasks=tasks,
        output_dir=output_dir,
    )

    return {
        "component_predictions": str(output_dir / "rmtgb_component_predictions.csv"),
        "sharing_diagnostics": str(output_dir / "rmtgb_sharing_diagnostics.csv"),
        "gating_weights": (
            str(output_dir / "rmtgb_gating_weights.csv")
            if (output_dir / "rmtgb_gating_weights.csv").exists()
            else None
        ),
    }


def save_gnn_diagnostics(model, tasks, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "adjacency_raw_"):
        raise AttributeError(
            "GNN does not expose adjacency_raw_. Add the fitted graph matrix "
            "to GraphVolatilityRegressor as self.adjacency_raw_."
        )

    raw = np.asarray(model.adjacency_raw_, dtype=float)

    if hasattr(model, "adjacency_"):
        normalized = np.asarray(model.adjacency_, dtype=float)
    else:
        normalized = raw.copy()

    if hasattr(model, "task_ids_"):
        task_ids = [int(v) for v in np.asarray(model.task_ids_).reshape(-1)]
    elif hasattr(model, "task_values_"):
        task_ids = [int(v) for v in np.asarray(model.task_values_).reshape(-1)]
    else:
        task_ids = list(range(raw.shape[0]))

    if raw.shape != (len(task_ids), len(task_ids)):
        raise ValueError(
            f"adjacency_raw_ has shape {raw.shape}, but {len(task_ids)} task IDs were found."
        )

    labels = [_safe_task_name(t, tasks) for t in task_ids]

    pd.DataFrame(raw, index=labels, columns=labels).to_csv(
        output_dir / "gnn_adjacency_raw.csv"
    )
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(
        output_dir / "gnn_adjacency_normalized.csv"
    )

    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append(
                {
                    "source_task_id": task_ids[i],
                    "source": labels[i],
                    "target_task_id": task_ids[j],
                    "target": labels[j],
                    "edge_weight": float(raw[i, j]),
                    "normalized_weight": float(normalized[i, j]),
                    "active": bool(raw[i, j] > 0.0),
                }
            )

    edge_df = pd.DataFrame(rows)
    edge_df = edge_df.sort_values(
        ["edge_weight", "normalized_weight"],
        ascending=False,
    ).reset_index(drop=True)
    edge_df.insert(0, "rank", np.arange(1, len(edge_df) + 1))

    edge_df.to_csv(
        output_dir / "gnn_edge_weights.csv",
        index=False,
    )

    node_rows = []
    for i, label in enumerate(labels):
        off_diag = np.delete(raw[i], i)
        active = off_diag > 0.0
        node_rows.append(
            {
                "task_id": task_ids[i],
                "task": label,
                "weighted_degree": float(np.sum(off_diag)),
                "mean_edge_weight": float(np.mean(off_diag)),
                "active_degree": int(np.sum(active)),
                "max_edge_weight": float(np.max(off_diag)) if len(off_diag) else 0.0,
            }
        )

    pd.DataFrame(node_rows).to_csv(
        output_dir / "gnn_node_connectivity.csv",
        index=False,
    )

    return {
        "raw_adjacency": str(output_dir / "gnn_adjacency_raw.csv"),
        "normalized_adjacency": str(output_dir / "gnn_adjacency_normalized.csv"),
        "edge_weights": str(output_dir / "gnn_edge_weights.csv"),
        "node_connectivity": str(output_dir / "gnn_node_connectivity.csv"),
    }


def save_dnn_mtl_diagnostics(model, X, y, tasks, output_dir):
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = (
        "x_scaler_",
        "task_map_",
        "model_",
        "device_",
        "y_mean_",
        "y_std_",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(
            f"DNN-MTL is missing fitted attributes required for diagnostics: {missing}"
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    raw_task = X[:, -1].astype(int)

    x_scaled = model.x_scaler_.transform(X[:, :-1]).astype("float32")
    model.model_.eval()

    with torch.no_grad():
        xt = torch.tensor(
            x_scaled,
            dtype=torch.float32,
            device=model.device_,
        )
        latent = model.model_.shared(xt).detach().cpu().numpy()

    full_pred = _to_1d_prediction(model.predict(X.copy()))

    task_ids = sorted(np.unique(raw_task))
    labels = [_safe_task_name(t, tasks) for t in task_ids]

    reliance_rows = []
    centroids = []
    head_weights = []

    for tid in task_ids:
        mask = raw_task == tid
        internal_id = int(model.task_map_[int(tid)])
        head = model.model_.heads[internal_id]

        bias_std = float(head.bias.detach().cpu().numpy().reshape(-1)[0])
        bias_original = bias_std * float(model.y_std_) + float(model.y_mean_)
        bias_only = np.full(mask.sum(), bias_original, dtype=float)

        full_rmse = _rmse(y[mask], full_pred[mask])
        bias_rmse = _rmse(y[mask], bias_only)

        gain = bias_rmse - full_rmse
        relative = gain / max(bias_rmse, 1e-12)

        task_latent = latent[mask]
        centroid = np.mean(task_latent, axis=0)
        centroids.append(centroid)

        weight = head.weight.detach().cpu().numpy().reshape(-1)
        head_weights.append(weight)

        reliance_rows.append(
            {
                "task_id": int(tid),
                "task": _safe_task_name(tid, tasks),
                "rmse_full": full_rmse,
                "rmse_bias_only": bias_rmse,
                "shared_representation_rmse_gain": gain,
                "relative_shared_reliance": relative,
                "latent_mean_norm": float(np.mean(np.linalg.norm(task_latent, axis=1))),
                "latent_centroid_norm": float(np.linalg.norm(centroid)),
                "head_weight_norm": float(np.linalg.norm(weight)),
            }
        )

    reliance_df = pd.DataFrame(reliance_rows)
    reliance_df.to_csv(
        output_dir / "dnn_shared_representation_reliance.csv",
        index=False,
    )

    centroids = np.vstack(centroids)
    pd.DataFrame(centroids, index=labels).to_csv(
        output_dir / "dnn_latent_centroids.csv"
    )

    latent_similarity = _cosine_similarity_matrix(centroids)
    pd.DataFrame(
        latent_similarity,
        index=labels,
        columns=labels,
    ).to_csv(output_dir / "dnn_latent_centroid_similarity.csv")

    head_weights = np.vstack(head_weights)
    pd.DataFrame(head_weights, index=labels).to_csv(
        output_dir / "dnn_task_head_weights.csv"
    )

    head_similarity = _cosine_similarity_matrix(head_weights)
    pd.DataFrame(
        head_similarity,
        index=labels,
        columns=labels,
    ).to_csv(output_dir / "dnn_task_head_similarity.csv")

    pair_rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair_rows.append(
                {
                    "task_i": labels[i],
                    "task_j": labels[j],
                    "latent_centroid_cosine": float(latent_similarity[i, j]),
                    "task_head_cosine": float(head_similarity[i, j]),
                }
            )

    pd.DataFrame(pair_rows).sort_values(
        "latent_centroid_cosine",
        ascending=False,
    ).to_csv(
        output_dir / "dnn_pairwise_similarity.csv",
        index=False,
    )

    return {
        "shared_reliance": str(output_dir / "dnn_shared_representation_reliance.csv"),
        "latent_centroids": str(output_dir / "dnn_latent_centroids.csv"),
        "latent_similarity": str(output_dir / "dnn_latent_centroid_similarity.csv"),
        "head_weights": str(output_dir / "dnn_task_head_weights.csv"),
        "head_similarity": str(output_dir / "dnn_task_head_similarity.csv"),
        "pairwise_similarity": str(output_dir / "dnn_pairwise_similarity.csv"),
    }


def save_model_interpretability(
    key,
    model,
    X_test,
    y_test,
    tasks,
    model_dir,
):
    diagnostics_dir = Path(model_dir) / "interpretability"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if key == "rmtgb":
        return save_rmtgb_diagnostics(
            model=model,
            X=X_test,
            y=y_test,
            tasks=tasks,
            output_dir=diagnostics_dir,
        )

    if key == "gnn":
        return save_gnn_diagnostics(
            model=model,
            tasks=tasks,
            output_dir=diagnostics_dir,
        )

    if key == "dnn_mtl":
        return save_dnn_mtl_diagnostics(
            model=model,
            X=X_test,
            y=y_test,
            tasks=tasks,
            output_dir=diagnostics_dir,
        )

    return {}
