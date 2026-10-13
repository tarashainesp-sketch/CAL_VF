from __future__ import annotations

import numpy as np
import pandas as pd


def _moving_block_indices(
    n_obs,
    block_length,
    rng,
):
    n_obs = int(n_obs)
    block_length = max(
        1,
        min(
            int(block_length),
            n_obs,
        ),
    )

    n_blocks = int(np.ceil(n_obs / block_length))

    max_start = max(
        1,
        n_obs - block_length + 1,
    )

    starts = rng.integers(
        0,
        max_start,
        size=n_blocks,
    )

    indices = []

    for start in starts:
        indices.extend(
            range(
                int(start),
                min(
                    int(start) + block_length,
                    n_obs,
                ),
            )
        )

    if len(indices) < n_obs:
        indices.extend(
            range(
                0,
                n_obs - len(indices),
            )
        )

    return np.asarray(
        indices[:n_obs],
        dtype=int,
    )


def _max_statistic(losses):
    losses = np.asarray(
        losses,
        dtype=float,
    )

    mean_losses = losses.mean(axis=0)

    centered = mean_losses - mean_losses.mean()

    return float(np.max(np.abs(centered)))


def model_confidence_set(
    losses,
    model_names,
    alpha=0.10,
    n_bootstrap=1000,
    block_length=4,
    random_state=42,
):
    losses = np.asarray(
        losses,
        dtype=float,
    )

    if losses.ndim != 2:
        raise ValueError("losses must have shape " "(n_observations, n_models).")

    if losses.shape[1] != len(model_names):
        raise ValueError("model_names must match " "the number of loss columns.")

    if losses.shape[0] < 3:
        raise ValueError("At least three observations " "are required for MCS.")

    if not np.all(np.isfinite(losses)):
        raise ValueError("MCS losses contain non-finite values.")

    rng = np.random.default_rng(int(random_state))

    active = list(range(losses.shape[1]))

    records = {
        i: {
            "model": model_names[i],
            "included": True,
            "elimination_step": np.nan,
            "p_value": np.nan,
            "mean_loss": float(losses[:, i].mean()),
        }
        for i in active
    }

    step = 0

    while len(active) > 1:
        step += 1

        current = losses[
            :,
            active,
        ]

        observed = _max_statistic(current)

        centered = current - current.mean(
            axis=0,
            keepdims=True,
        )

        bootstrap_stats = np.empty(
            int(n_bootstrap),
            dtype=float,
        )

        for b in range(int(n_bootstrap)):
            idx = _moving_block_indices(
                current.shape[0],
                block_length,
                rng,
            )

            boot = centered[idx, :]

            bootstrap_stats[b] = _max_statistic(boot)

        p_value = float(
            (1.0 + np.sum(bootstrap_stats >= observed)) / (int(n_bootstrap) + 1.0)
        )

        if p_value >= float(alpha):
            for i in active:
                if np.isnan(records[i]["p_value"]):
                    records[i]["p_value"] = p_value

            break

        active_mean = current.mean(axis=0)

        worst_local = int(np.argmax(active_mean))

        worst_global = active[worst_local]

        records[worst_global]["included"] = False

        records[worst_global]["elimination_step"] = step

        records[worst_global]["p_value"] = p_value

        active.remove(worst_global)

    for i in active:
        records[i]["included"] = True

    result = pd.DataFrame([records[i] for i in range(len(model_names))])

    result["alpha"] = float(alpha)
    result["n_bootstrap"] = int(n_bootstrap)
    result["block_length"] = int(block_length)

    return result


def build_loss_matrix(
    prediction_df,
    loss_name,
    task=None,
):
    required = {
        "date",
        "task",
        "model_key",
        "model",
        "y_true",
        "y_pred",
    }

    missing = required.difference(prediction_df.columns)

    if missing:
        raise ValueError(
            "Prediction data are missing columns: " + ", ".join(sorted(missing))
        )

    df = prediction_df.copy()

    if task is not None:
        df = df[df["task"] == task].copy()

    error = df["y_true"].to_numpy(dtype=float) - df["y_pred"].to_numpy(dtype=float)

    if loss_name == "SE":
        df["loss"] = error**2
    elif loss_name == "AE":
        df["loss"] = np.abs(error)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    if task is None:
        collapsed = df.groupby(
            [
                "date",
                "model_key",
                "model",
            ],
            as_index=False,
        )["loss"].mean()
    else:
        collapsed = df[
            [
                "date",
                "model_key",
                "model",
                "loss",
            ]
        ].copy()

    labels = (
        collapsed[
            [
                "model_key",
                "model",
            ]
        ]
        .drop_duplicates()
        .sort_values("model_key")
        .reset_index(drop=True)
    )

    matrix = collapsed.pivot(
        index="date",
        columns="model_key",
        values="loss",
    ).sort_index()

    ordered_keys = [
        key for key in labels["model_key"].tolist() if key in matrix.columns
    ]

    matrix = matrix[ordered_keys].dropna(
        axis=0,
        how="any",
    )

    names = [
        labels.loc[
            labels["model_key"] == key,
            "model",
        ].iloc[0]
        for key in ordered_keys
    ]

    return (
        matrix.to_numpy(dtype=float),
        ordered_keys,
        names,
        matrix.index,
    )


def run_mcs_suite(
    prediction_df,
    tasks,
    alpha=0.10,
    n_bootstrap=1000,
    block_length=4,
    random_state=42,
):
    outputs = []

    task_values = [
        "ALL",
        *[tasks[k] for k in sorted(tasks)],
    ]

    for loss_name in [
        "SE",
        "AE",
    ]:
        for task_name in task_values:
            selected_task = None if task_name == "ALL" else task_name

            losses, keys, names, _ = build_loss_matrix(
                prediction_df,
                loss_name,
                task=selected_task,
            )

            result = model_confidence_set(
                losses,
                names,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                block_length=block_length,
                random_state=(
                    int(random_state) + sum(ord(c) for c in (loss_name + task_name))
                ),
            )

            result.insert(
                0,
                "model_key",
                keys,
            )

            result.insert(
                0,
                "loss",
                loss_name,
            )

            result.insert(
                0,
                "task",
                task_name,
            )

            outputs.append(result)

    return pd.concat(
        outputs,
        ignore_index=True,
    )
