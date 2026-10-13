from __future__ import annotations

import copy
import random

import numpy as np
from sklearn.preprocessing import StandardScaler


class GraphVolatilityRegressor:
    def __init__(
        self,
        hidden_dim=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        epochs=150,
        patience=25,
        validation_fraction=0.15,
        rv_col=2,
        graph_threshold=0.0,
        random_state=42,
        device=None,
    ):
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.rv_col = int(rv_col)
        self.graph_threshold = float(graph_threshold)
        self.random_state = int(random_state)
        self.device = device

    def _require_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("GraphVolatilityRegressor requires PyTorch.") from exc
        return torch, nn

    @staticmethod
    def _panelize(X, y=None):
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        data = X[:, :-1]
        task_ids = [int(v) for v in np.unique(task)]
        indices = {t: np.flatnonzero(task == t) for t in task_ids}
        lengths = [len(indices[t]) for t in task_ids]
        if len(set(lengths)) != 1:
            raise ValueError("GNN requires a balanced task panel in each split.")
        panel_x = np.stack([data[indices[t]] for t in task_ids], axis=1)
        panel_y = None
        if y is not None:
            y = np.asarray(y, dtype=float).reshape(-1)
            panel_y = np.stack([y[indices[t]] for t in task_ids], axis=1)
        return panel_x, panel_y, task_ids, indices

    @staticmethod
    def _normalize_adjacency(a):
        a = np.asarray(a, dtype=float)
        a = (a + a.T) / 2.0
        a = a + np.eye(a.shape[0])
        d = np.sum(a, axis=1)
        inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        return inv[:, None] * a * inv[None, :]

    def _make_model(self, n_features, n_tasks, adjacency):
        torch, nn = self._require_torch()

        class Net(nn.Module):
            def __init__(self, n_features, hidden_dim, adjacency):
                super().__init__()
                self.register_buffer(
                    "adj", torch.tensor(adjacency, dtype=torch.float32)
                )
                self.lin1 = nn.Linear(n_features, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, hidden_dim)
                self.head = nn.Linear(hidden_dim, 1)
                self.act = nn.ReLU()

            def graph_mix(self, x):
                return torch.einsum("ij,bjf->bif", self.adj, x)

            def forward(self, x):
                h = self.act(self.lin1(self.graph_mix(x)))
                h = self.act(self.lin2(self.graph_mix(h)))
                return self.head(h).squeeze(-1)

        return Net(n_features, self.hidden_dim, adjacency)

    def fit(self, X, y):
        torch, nn = self._require_torch()
        panel_x, panel_y, task_ids, _ = self._panelize(X, y)
        self.task_ids_ = task_ids
        self.task_to_pos_ = {t: i for i, t in enumerate(task_ids)}

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        n_time, n_tasks, n_features = panel_x.shape
        flat_x = panel_x.reshape(-1, n_features)
        self.x_scaler_ = StandardScaler().fit(flat_x)
        x = (
            self.x_scaler_.transform(flat_x)
            .reshape(n_time, n_tasks, n_features)
            .astype(np.float32)
        )
        self.y_mean_ = float(np.mean(panel_y))
        self.y_std_ = max(float(np.std(panel_y, ddof=1)), 1e-8)
        ys = ((panel_y - self.y_mean_) / self.y_std_).astype(np.float32)

        rv_panel = panel_x[:, :, self.rv_col]
        corr = np.corrcoef(rv_panel, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        if self.graph_threshold > 0:
            corr[corr < self.graph_threshold] = 0.0
        self.adjacency_raw_ = corr.copy()
        self.adjacency_ = self._normalize_adjacency(corr)

        n_val = max(1, int(np.ceil(n_time * self.validation_fraction)))
        n_train = n_time - n_val
        if n_train < 10:
            raise ValueError("Too few GNN training weeks after validation split.")

        tx, vx = x[:n_train], x[n_train:]
        ty, vy = ys[:n_train], ys[n_train:]
        dataset = torch.utils.data.TensorDataset(torch.tensor(tx), torch.tensor(ty))
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        self.device_ = torch.device(
            self.device
            if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_ = self._make_model(n_features, n_tasks, self.adjacency_).to(
            self.device_
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        vx_t = torch.tensor(vx, device=self.device_)
        vy_t = torch.tensor(vy, device=self.device_)
        best_loss = np.inf
        best_state = None
        bad = 0
        self.history_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            batch_losses = []
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(vx_t), vy_t).item())
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(batch_losses)),
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = float(best_loss)
        return self

    def predict(self, X):
        torch, _ = self._require_torch()
        panel_x, _, task_ids, indices = self._panelize(X, None)
        if task_ids != self.task_ids_:
            raise ValueError(
                "Prediction task set/order differs from GNN training task set."
            )
        n_time, n_tasks, n_features = panel_x.shape
        x = (
            self.x_scaler_.transform(panel_x.reshape(-1, n_features))
            .reshape(n_time, n_tasks, n_features)
            .astype(np.float32)
        )

        self.model_.eval()
        with torch.no_grad():
            panel_pred = (
                self.model_(torch.tensor(x, dtype=torch.float32, device=self.device_))
                .cpu()
                .numpy()
            )
        panel_pred = np.maximum(panel_pred * self.y_std_ + self.y_mean_, 0.0)

        pred = np.empty(len(X), dtype=float)
        for j, task_id in enumerate(task_ids):
            pred[indices[task_id]] = panel_pred[:, j]
        return pred

    def get_training_history(self):
        return list(getattr(self, "history_", []))
