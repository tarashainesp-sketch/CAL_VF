from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.dnn_mtl import DNNMTLRegressor
from models.har_rv import WeeklyHARRVRegressor
from models.garch import GARCHRegressor
from models.realized_garch import RealizedGARCHRegressor
from models.dcc_heavy import DCCHEAVYRegressor
from models.gnn_vol import GraphVolatilityRegressor
from models.lstm_vol import PerTaskLSTMRegressor

from utils.bo_search import BO_MODEL_KEYS, bayes_search
from utils.data import build_weekly_ml_from_yfinance, to_xy
from utils.mcs import run_mcs_suite
from utils.interpretability import save_model_interpretability
from utils.demsar import save_seed_demsar_inputs, aggregate_demsar_inputs

TASKS = {
    0: "SPY",
    1: "QQQ",
    2: "IWM",
    3: "EEM",
    4: "TLT",
    5: "IEF",
    6: "LQD",
    7: "HYG",
    8: "GLD",
    9: "SLV",
    10: "USO",
    11: "DBA",
    12: "BTC-USD",
    13: "ETH-USD",
}

TASK_SYMBOLS = dict(TASKS)


FIXED = {
    "har_rv": {},
    "gb": {
        "max_iter": 150,
        "learning_rate": 0.05,
        "max_leaf_nodes": 2,
        "l2_regularization": 0.0,
    },
    "garch": {
        "return_col": 0,
        "maxiter": 1000,
    },
    "realized_garch": {
        "return_col": 0,
        "rv_col": 2,
        "maxiter": 1500,
    },
    "dcc_heavy": {
        "return_col": 0,
        "rv_col": 2,
        "maxiter": 1000,
    },
    "lstm": {
        "hidden_dim": 32,
        "num_layers": 1,
        "lookback": 8,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 150,
        "patience": 25,
        "validation_fraction": 0.15,
    },
    "gnn": {
        "hidden_dim": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 150,
        "patience": 25,
        "validation_fraction": 0.15,
        "rv_col": 2,
        "graph_threshold": 0.15,
    },
    "rmtgb": {
        "n_iter_1st": 60,
        "n_iter_2nd": 30,
        "n_iter_3rd": 60,
        "max_depth": 1,
        "subsample": 1.0,
        "learning_rate": 0.05,
    },
    "dnn_mtl": {
        "hidden1": 100,
        "hidden2": 100,
        "hidden3": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "epochs": 100,
        "patience": 100,
        "validation_fraction": 0.15,
    },
}


MODEL_ORDER = [
    "har_rv",
    "garch",
    "realized_garch",
    "dcc_heavy",
    "gb",
    "lstm",
    "gnn",
    "rmtgb",
    "dnn_mtl",
]


def make_balanced_calendar_panel(weekly, n_tasks):
    df = weekly.copy()

    if "date" not in df.columns:
        raise ValueError("weekly dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])

    task_col = None
    for candidate in ("task_id", "task", "ticker", "symbol"):
        if candidate in df.columns:
            task_col = candidate
            break

    if task_col is None:
        counts = df.groupby("date").size()
        good_dates = counts[counts == int(n_tasks)].index
    else:
        counts = df.groupby("date")[task_col].nunique()
        good_dates = counts[counts == int(n_tasks)].index

    df = df[df["date"].isin(good_dates)].copy()

    if task_col is not None:
        dup = df.duplicated(["date", task_col], keep=False)
        if dup.any():
            raise ValueError(
                "Balanced panel construction found duplicate task/date rows."
            )

    per_date = df.groupby("date").size()
    if len(df) == 0 or not np.all(per_date.to_numpy() == int(n_tasks)):
        raise ValueError(
            f"Could not construct a complete {n_tasks}-task calendar panel."
        )

    sort_cols = ["date"] + ([task_col] if task_col is not None else [])

    return df.sort_values(sort_cols).reset_index(drop=True)


def balanced_calendar_split(
    df,
    train_frac=0.60,
    val_frac=0.20,
):
    dates = np.array(sorted(pd.to_datetime(df["date"]).unique()))

    n_dates = len(dates)

    if n_dates < 20:
        raise ValueError(f"Too few balanced weekly dates: {n_dates}.")

    n_train = max(
        int(np.floor(n_dates * float(train_frac))),
        1,
    )

    n_val = max(
        int(np.floor(n_dates * float(val_frac))),
        1,
    )

    if n_train + n_val >= n_dates:
        raise ValueError("Invalid calendar split: no dates left for the test set.")

    train_dates = set(dates[:n_train])

    val_dates = set(dates[n_train : n_train + n_val])

    test_dates = set(dates[n_train + n_val :])

    d = pd.to_datetime(df["date"])

    train = df[d.isin(train_dates)].copy().reset_index(drop=True)

    val = df[d.isin(val_dates)].copy().reset_index(drop=True)

    test = df[d.isin(test_dates)].copy().reset_index(drop=True)

    return train, val, test


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, np.datetime64):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


class PerTaskGB:
    def __init__(
        self,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=2,
        l2_regularization=0.0,
        random_state=42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        task = X[:, -1].astype(int)
        data = X[:, :-1]

        self.models_ = {}

        for task_id in np.unique(task):
            mask = task == task_id

            model = HistGradientBoostingRegressor(
                max_iter=int(self.max_iter),
                learning_rate=float(self.learning_rate),
                max_leaf_nodes=int(self.max_leaf_nodes),
                l2_regularization=float(self.l2_regularization),
                random_state=int(self.random_state),
            )

            model.fit(
                data[mask],
                y[mask],
            )

            self.models_[int(task_id)] = model

        return self

    def predict(self, X):
        X = np.asarray(X)

        task = X[:, -1].astype(int)
        data = X[:, :-1]

        pred = np.empty(
            len(X),
            dtype=float,
        )

        for task_id in np.unique(task):
            mask = task == task_id

            pred[mask] = self.models_[int(task_id)].predict(data[mask])

        return pred


def require_rmtgb():
    try:
        from models.rmtgb.mtgb import MTGBRegressor
    except Exception as exc:
        raise ImportError(
            "R-MTGB requires the original MTGBRegressor "
            "implementation in models/rmtgb/."
        ) from exc

    return MTGBRegressor


def factories(seed):
    seed = int(seed)

    MTGBRegressor = require_rmtgb()

    return {
        "har_rv": (
            "HAR-RV (independent)",
            lambda **params: WeeklyHARRVRegressor(**params),
        ),
        "garch": (
            "GARCH(1,1)",
            lambda **params: GARCHRegressor(
                random_state=seed,
                **params,
            ),
        ),
        "realized_garch": (
            "Realized GARCH",
            lambda **params: RealizedGARCHRegressor(
                random_state=seed,
                **params,
            ),
        ),
        "dcc_heavy": (
            "DCC-HEAVY (marginal)",
            lambda **params: DCCHEAVYRegressor(
                random_state=seed,
                **params,
            ),
        ),
        "gb": (
            "HGB (independent)",
            lambda **params: PerTaskGB(
                random_state=seed,
                **params,
            ),
        ),
        "lstm": (
            "LSTM (independent)",
            lambda **params: PerTaskLSTMRegressor(
                random_state=seed,
                **params,
            ),
        ),
        "gnn": (
            "GNN (cross-market)",
            lambda **params: GraphVolatilityRegressor(
                random_state=seed,
                **params,
            ),
        ),
        "rmtgb": (
            "R-MTGB (adaptive MTL)",
            lambda **params: MTGBRegressor(
                random_state=seed,
                **params,
            ),
        ),
        "dnn_mtl": (
            "DNN-MTL",
            lambda **params: DNNMTLRegressor(
                random_state=seed,
                **params,
            ),
        ),
    }


def seed_everything(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    except ImportError:
        pass


def rmse_objective(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_row(
    model_key,
    model_name,
    task_name,
    task_id,
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    err = y_true - y_pred

    rmse = float(np.sqrt(np.mean(err**2)))

    mae = float(np.mean(np.abs(err)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))

    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else np.nan

    return {
        "model_key": model_key,
        "model": model_name,
        "task": task_name,
        "task_id": task_id,
        "n": int(len(y_true)),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def evaluate_model(
    model_key,
    model_name,
    y_true,
    y_pred,
    task_id,
):
    rows = []

    rows.append(
        metric_row(
            model_key,
            model_name,
            "ALL",
            np.nan,
            y_true,
            y_pred,
        )
    )

    per_task = []

    for tid in sorted(np.unique(task_id)):
        mask = task_id == tid

        row = metric_row(
            model_key,
            model_name,
            TASKS[int(tid)],
            int(tid),
            y_true[mask],
            y_pred[mask],
        )

        rows.append(row)

        per_task.append(row)

    rows.append(
        {
            "model_key": model_key,
            "model": model_name,
            "task": "TASK_AVG",
            "task_id": np.nan,
            "n": int(len(y_true)),
            "RMSE": float(np.mean([r["RMSE"] for r in per_task])),
            "MAE": float(np.mean([r["MAE"] for r in per_task])),
            "R2": float(np.mean([r["R2"] for r in per_task])),
        }
    )

    return pd.DataFrame(rows)


def prediction_frame(
    model_key,
    model_name,
    y_true,
    y_pred,
    X_test,
    test_df,
):
    task_id = np.asarray(
        X_test[
            :,
            -1,
        ],
        dtype=int,
    )

    if "date" in test_df.columns:
        dates = pd.to_datetime(test_df["date"]).astype(str).to_numpy()
    else:
        dates = np.arange(len(y_true)).astype(str)

    return pd.DataFrame(
        {
            "date": dates,
            "task_id": task_id,
            "task": [TASKS[int(v)] for v in task_id],
            "model_key": model_key,
            "model": model_name,
            "y_true": np.asarray(
                y_true,
                dtype=float,
            ),
            "y_pred": np.asarray(
                y_pred,
                dtype=float,
            ),
        }
    )


def model_diagnostics(model):
    keep = [
        "params_",
        "converged_",
        "dcc_params_",
        "dcc_converged_",
        "panel_lengths_",
        "dcc_panel_length_",
        "dcc_panel_was_balanced_",
        "adjacency_raw_",
        "adjacency_",
        "best_val_loss_",
        "theta",
        "theta_",
    ]

    return {
        name: json_safe(
            getattr(
                model,
                name,
            )
        )
        for name in keep
        if hasattr(
            model,
            name,
        )
    }


def run_seed(
    data_path,
    seed,
    use_bo,
    bo_calls,
    results_root,
    mcs_alpha,
    mcs_bootstrap,
    mcs_block_length,
):
    seed = int(seed)

    seed_everything(seed)

    results_root = Path(results_root)

    seed_dir = results_root / f"seed_{seed}"

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly, feature_cols = build_weekly_ml_from_yfinance(
        data_path,
        TASK_SYMBOLS,
        price_field="Adj Close",
        n_rv_lags=4,
    )

    weekly.to_csv(
        seed_dir / "weekly_ml_dataset_unbalanced.csv",
        index=False,
    )

    weekly = make_balanced_calendar_panel(
        weekly,
        len(TASKS),
    )

    weekly.to_csv(
        seed_dir / "weekly_ml_dataset.csv",
        index=False,
    )

    n_balanced_dates = int(pd.to_datetime(weekly["date"]).nunique())

    train_df, val_df, test_df = balanced_calendar_split(
        weekly,
        0.60,
        0.20,
    )

    X_train, y_train = to_xy(
        train_df,
        feature_cols,
    )

    X_val, y_val = to_xy(
        val_df,
        feature_cols,
    )

    X_test, y_test = to_xy(
        test_df,
        feature_cols,
    )

    fit_df = pd.concat(
        [
            train_df,
            val_df,
        ],
        ignore_index=True,
    )

    X_fit, y_fit = to_xy(
        fit_df,
        feature_cols,
    )

    model_factories = factories(seed)

    all_metrics = []
    all_predictions = []

    meta = {
        "seed": seed,
        "tasks": {str(k): v for k, v in TASKS.items()},
        "feature_columns": list(feature_cols),
        "use_bayesian_optimization": bool(use_bo),
        "bo_calls": int(bo_calls),
        "sample": {
            "balanced_dates": n_balanced_dates,
            "train_dates": int(pd.to_datetime(train_df["date"]).nunique()),
            "validation_dates": int(pd.to_datetime(val_df["date"]).nunique()),
            "test_dates": int(pd.to_datetime(test_df["date"]).nunique()),
        },
        "models": {},
    }

    for key in MODEL_ORDER:
        name, factory = model_factories[key]

        model_dir = seed_dir / key

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        params = dict(FIXED[key])

        tuning = {
            "enabled": False,
            "fixed_parameters": json_safe(params),
        }

        if use_bo and key in BO_MODEL_KEYS:
            best, history = bayes_search(
                key,
                factory,
                X_train,
                y_train,
                X_val,
                y_val,
                rmse_objective,
                bo_calls,
                seed,
            )

            params.update(best)

            history.to_csv(
                model_dir / "bo_history.csv",
                index=False,
            )

            with open(
                model_dir / "bo_best_params.json",
                "w",
            ) as handle:
                json.dump(
                    json_safe(best),
                    handle,
                    indent=2,
                )

            objective = history["validation_objective"].to_numpy(dtype=float)

            valid_mask = np.isfinite(objective) & (objective < 1e12)

            valid = history[valid_mask]

            if valid.empty:
                errors = (
                    history["error"].dropna().value_counts().head(5).to_dict()
                    if "error" in history.columns
                    else {}
                )

                raise RuntimeError(
                    f"Bayesian optimization "
                    f"failed for {key}. "
                    f"All {len(history)} "
                    f"evaluations failed. "
                    f"Most common errors: "
                    f"{errors}. "
                    f"See "
                    f"{model_dir / 'bo_history.csv'}."
                )

            tuning = {
                "enabled": True,
                "n_calls": int(len(history)),
                "best_parameters": json_safe(best),
                "best_validation_objective": float(valid["validation_objective"].min()),
            }

        model = factory(**params)

        model.fit(
            X_fit.copy(),
            y_fit,
        )

        pred = np.asarray(
            model.predict(X_test.copy()),
            dtype=float,
        ).reshape(-1)

        pred = np.maximum(
            pred,
            0.0,
        )

        task_id = np.asarray(
            X_test[
                :,
                -1,
            ],
            dtype=int,
        )

        metrics = evaluate_model(
            key,
            name,
            np.asarray(y_test),
            pred,
            task_id,
        )

        all_metrics.append(metrics)

        pred_df = prediction_frame(
            key,
            name,
            y_test,
            pred,
            X_test,
            test_df,
        )

        pred_df.to_csv(
            model_dir / "test_predictions.csv",
            index=False,
        )

        all_predictions.append(pred_df)

        diagnostics = model_diagnostics(model)

        with open(
            model_dir / "model_diagnostics.json",
            "w",
        ) as handle:
            json.dump(
                json_safe(diagnostics),
                handle,
                indent=2,
            )

        interpretability_meta = {
            "saved": False,
            "files": {},
            "error": None,
        }

        if key in {
            "rmtgb",
            "gnn",
            "dnn_mtl",
        }:
            try:
                saved = save_model_interpretability(
                    key=key,
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    tasks=TASKS,
                    model_dir=model_dir,
                )

                interpretability_meta = {
                    "saved": True,
                    "files": json_safe(saved),
                    "error": None,
                }

            except Exception as exc:
                interpretability_meta = {
                    "saved": False,
                    "files": {},
                    "error": repr(exc),
                }

        model_meta = {
            "name": name,
            "parameters": json_safe(params),
            "tuning": json_safe(tuning),
            "diagnostics": json_safe(diagnostics),
            "interpretability": json_safe(interpretability_meta),
        }

        meta["models"][key] = model_meta

        with open(
            model_dir / "model_metadata.json",
            "w",
        ) as handle:
            json.dump(
                json_safe(model_meta),
                handle,
                indent=2,
            )

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    metrics_df.to_csv(
        seed_dir / "model_comparison.csv",
        index=False,
    )

    metrics_df[
        metrics_df["task"].isin(
            [
                "ALL",
                "TASK_AVG",
            ]
        )
    ].to_csv(
        seed_dir / "model_summary.csv",
        index=False,
    )

    save_seed_demsar_inputs(
        metrics_df=metrics_df,
        seed=seed,
        output_dir=seed_dir,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        seed_dir / "all_model_predictions.csv",
        index=False,
    )

    mcs_results = run_mcs_suite(
        predictions_df,
        TASKS,
        alpha=float(mcs_alpha),
        n_bootstrap=int(mcs_bootstrap),
        block_length=int(mcs_block_length),
        random_state=seed,
    )

    mcs_results.insert(
        0,
        "seed",
        seed,
    )

    mcs_results.to_csv(
        seed_dir / "mcs_results.csv",
        index=False,
    )

    try:
        dem_outputs = aggregate_demsar_inputs(results_root)

        meta["demsar_outputs"] = json_safe(dem_outputs)

    except Exception as exc:
        meta["demsar_outputs"] = {"error": repr(exc)}

    with open(
        seed_dir / "experiment_metadata.json",
        "w",
    ) as handle:
        json.dump(
            json_safe(meta),
            handle,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description=("Run the revised cross-asset " "volatility benchmark suite.")
    )

    parser.add_argument(
        "--data",
        default=("data/" "cross_asset_daily_data.csv"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--bo",
        dest="use_bo",
        action="store_true",
    )

    mode.add_argument(
        "--no-bo",
        dest="use_bo",
        action="store_false",
    )

    parser.set_defaults(use_bo=False)

    parser.add_argument(
        "--bo-calls",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--results-root",
        default="results",
    )

    parser.add_argument(
        "--mcs-alpha",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--mcs-bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--mcs-block-length",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    run_seed(
        data_path=data_path,
        seed=args.seed,
        use_bo=args.use_bo,
        bo_calls=args.bo_calls,
        results_root=args.results_root,
        mcs_alpha=args.mcs_alpha,
        mcs_bootstrap=args.mcs_bootstrap,
        mcs_block_length=args.mcs_block_length,
    )


if __name__ == "__main__":
    main()
