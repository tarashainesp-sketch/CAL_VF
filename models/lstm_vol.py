from __future__ import annotations

import copy
import random

import numpy as np
from sklearn.preprocessing import StandardScaler


class _TaskLSTMWrapper:
    pass


class PerTaskLSTMRegressor:

    def __init__(
        self,
        hidden_dim=32,
        num_layers=1,
        lookback=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        epochs=150,
        patience=25,
        validation_fraction=0.15,
        random_state=42,
        device=None,
    ):
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lookback = int(lookback)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.random_state = int(random_state)
        self.device = device

    def _require_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("PerTaskLSTMRegressor requires PyTorch.") from exc
        return torch, nn

    def _make_model(self, n_features):
        torch, nn = self._require_torch()

        class Net(nn.Module):
            def __init__(self, n_features, hidden_dim, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(n_features, self.hidden_dim, self.num_layers)

    @staticmethod
    def _sequences(x, y, lookback):
        xs, ys = [], []
        for i in range(lookback - 1, len(x)):
            xs.append(x[i - lookback + 1 : i + 1])
            ys.append(y[i])
        if not xs:
            raise ValueError("Not enough observations for requested LSTM lookback.")
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def fit(self, X, y):
        torch, nn = self._require_torch()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        task = X[:, -1].astype(int)
        data = X[:, :-1]

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.device_ = torch.device(
            self.device
            if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models_ = {}
        self.x_scalers_ = {}
        self.y_stats_ = {}
        self.history_tail_ = {}
        self.training_history_ = {}

        for task_id in np.unique(task):
            key = int(task_id)
            mask = task == task_id
            x_raw = data[mask]
            y_raw = y[mask]
            if len(x_raw) < self.lookback + 15:
                raise ValueError(f"Insufficient LSTM observations for task {key}.")

            scaler = StandardScaler().fit(x_raw)
            x = scaler.transform(x_raw).astype(np.float32)
            y_mean = float(np.mean(y_raw))
            y_std = max(float(np.std(y_raw, ddof=1)), 1e-8)
            ys = ((y_raw - y_mean) / y_std).astype(np.float32)
            seq_x, seq_y = self._sequences(x, ys, self.lookback)

            n_val = max(1, int(np.ceil(len(seq_x) * self.validation_fraction)))
            n_train = len(seq_x) - n_val
            if n_train < 5:
                raise ValueError(f"Too few LSTM training sequences for task {key}.")

            train_x, val_x = seq_x[:n_train], seq_x[n_train:]
            train_y, val_y = seq_y[:n_train], seq_y[n_train:]
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(train_x), torch.tensor(train_y)
            )
            generator = torch.Generator().manual_seed(self.random_state + key)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True,
                generator=generator,
            )

            model = self._make_model(x.shape[1]).to(self.device_)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            loss_fn = nn.MSELoss()
            vx = torch.tensor(val_x, device=self.device_)
            vy = torch.tensor(val_y, device=self.device_)
            best_loss = np.inf
            best_state = None
            bad = 0
            history = []

            for epoch in range(self.epochs):
                model.train()
                batch_losses = []
                for xb, yb in loader:
                    xb = xb.to(self.device_)
                    yb = yb.to(self.device_)
                    optimizer.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(float(loss.item()))

                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(vx), vy).item())
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(np.mean(batch_losses)),
                        "val_loss": val_loss,
                    }
                )
                if val_loss < best_loss - 1e-8:
                    best_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            self.models_[key] = model
            self.x_scalers_[key] = scaler
            self.y_stats_[key] = (y_mean, y_std)
            self.history_tail_[key] = x[-(self.lookback - 1) :].copy()
            self.training_history_[key] = history

        return self

    def predict(self, X):
        torch, _ = self._require_torch()
        X = np.asarray(X, dtype=float)
        task = X[:, -1].astype(int)
        data = X[:, :-1]
        pred = np.empty(len(X), dtype=float)

        for task_id in np.unique(task):
            key = int(task_id)
            idx = np.flatnonzero(task == task_id)
            x_new = self.x_scalers_[key].transform(data[idx]).astype(np.float32)
            history = self.history_tail_[key].copy()
            model = self.models_[key]
            model.eval()
            y_mean, y_std = self.y_stats_[key]

            for local_i, row_idx in enumerate(idx):
                current = x_new[local_i : local_i + 1]
                seq = np.vstack([history, current])[-self.lookback :]
                if len(seq) < self.lookback:
                    pad = np.repeat(seq[:1], self.lookback - len(seq), axis=0)
                    seq = np.vstack([pad, seq])
                with torch.no_grad():
                    z = float(
                        model(
                            torch.tensor(
                                seq[None, :, :],
                                dtype=torch.float32,
                                device=self.device_,
                            )
                        )
                        .cpu()
                        .item()
                    )
                pred[row_idx] = max(z * y_std + y_mean, 0.0)
                history = np.vstack([history, current])[-(self.lookback - 1) :]

        return pred

    def get_training_history(self):
        return self.training_history_
