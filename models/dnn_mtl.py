from __future__ import annotations

import copy
import random

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class DNNMTLRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden1=100,
        hidden2=100,
        hidden3=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        epochs=100,
        patience=100,
        validation_fraction=0.15,
        random_state=42,
        device=None,
    ):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device

    def _seed(self):
        import torch

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

    def _build(self, p, n_tasks):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, p, n_tasks, h1, h2, h3):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(p, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, h3),
                    nn.ReLU(),
                )
                self.heads = nn.ModuleList([nn.Linear(h3, 1) for _ in range(n_tasks)])

            def forward(self, x, t):
                z = self.shared(x)
                out = torch.empty(
                    x.shape[0],
                    device=x.device,
                    dtype=z.dtype,
                )

                for k, head in enumerate(self.heads):
                    mask = t == k

                    if torch.any(mask):
                        out[mask] = head(z[mask]).squeeze(-1)

                return out

        return Net(
            p,
            n_tasks,
            int(self.hidden1),
            int(self.hidden2),
            int(self.hidden3),
        )

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        self._seed()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        x = X[:, :-1]
        task = X[:, -1].astype(int)

        self.task_values_ = np.sort(np.unique(task))
        self.task_map_ = {int(value): i for i, value in enumerate(self.task_values_)}

        task_dense = np.array(
            [self.task_map_[int(value)] for value in task],
            dtype=int,
        )

        n_val = max(
            1,
            int(round(len(y) * float(self.validation_fraction))),
        )
        cut = max(1, len(y) - n_val)

        x_train = x[:cut]
        x_val = x[cut:]
        task_train = task_dense[:cut]
        task_val = task_dense[cut:]
        y_train = y[:cut]
        y_val = y[cut:]

        self.x_scaler_ = StandardScaler().fit(x_train)

        x_train = self.x_scaler_.transform(x_train).astype("float32")
        x_val = self.x_scaler_.transform(x_val).astype("float32")

        self.y_mean_ = float(y_train.mean())
        self.y_std_ = float(y_train.std())

        if self.y_std_ == 0.0:
            self.y_std_ = 1.0

        y_train = ((y_train - self.y_mean_) / self.y_std_).astype("float32")

        y_val = ((y_val - self.y_mean_) / self.y_std_).astype("float32")

        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ = self._build(
            x.shape[1],
            len(self.task_values_),
        ).to(self.device_)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(task_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(int(self.batch_size), len(dataset)),
            shuffle=True,
        )

        x_val_tensor = torch.tensor(
            x_val,
            dtype=torch.float32,
            device=self.device_,
        )
        task_val_tensor = torch.tensor(
            task_val,
            dtype=torch.long,
            device=self.device_,
        )
        y_val_tensor = torch.tensor(
            y_val,
            dtype=torch.float32,
            device=self.device_,
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        loss_fn = nn.MSELoss()

        best_loss = np.inf
        best_state = None
        bad_epochs = 0
        self.history_ = []

        for epoch in range(int(self.epochs)):
            self.model_.train()
            train_losses = []

            for xb, tb, yb in loader:
                xb = xb.to(self.device_)
                tb = tb.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad()

                pred = self.model_(xb, tb)
                loss = loss_fn(pred, yb)

                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))

            self.model_.eval()

            with torch.no_grad():
                val_loss = float(
                    loss_fn(
                        self.model_(
                            x_val_tensor,
                            task_val_tensor,
                        ),
                        y_val_tensor,
                    ).item()
                )

            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_loss,
                }
            )

            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

                if bad_epochs >= int(self.patience):
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.best_val_loss_ = float(best_loss)
        self.n_epochs_ = len(self.history_)

        return self

    def predict(self, X):
        import torch

        X = np.asarray(X, dtype=float)

        x = self.x_scaler_.transform(X[:, :-1]).astype("float32")

        raw_task = X[:, -1].astype(int)

        task = np.array(
            [self.task_map_[int(value)] for value in raw_task],
            dtype=int,
        )

        self.model_.eval()

        with torch.no_grad():
            pred = (
                self.model_(
                    torch.tensor(
                        x,
                        dtype=torch.float32,
                        device=self.device_,
                    ),
                    torch.tensor(
                        task,
                        dtype=torch.long,
                        device=self.device_,
                    ),
                )
                .cpu()
                .numpy()
            )

        return pred * self.y_std_ + self.y_mean_

    def get_training_history(self):
        return list(
            getattr(
                self,
                "history_",
                [],
            )
        )
