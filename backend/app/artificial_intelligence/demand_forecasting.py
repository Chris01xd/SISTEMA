"""Demand forecasting: predicts near-horizon patient arrivals.

Model selection justified in docs/ARCHITECTURE.md section 7:
- LSTM (torch) as primary model for its temporal-dependency modeling of the
  non-stationary surge/decay arrival pattern typical of MCEs.
- XGBoost retained as a low-data-regime fallback / interpretable baseline.

Both models share the same feature contract (a sliding window of recent
arrival counts + saturation index) so they are interchangeable at inference
time via `ForecastModel.predict()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # torch is an optional heavy dependency for the API-only dev path
    torch = None
    nn = None

from xgboost import XGBRegressor

WINDOW_SIZE = 12  # last 12 timesteps


class ForecastModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...


class XGBoostDemandForecaster(ForecastModel):
    """Fallback / baseline model — robust with the small samples a single
    hospital's historical MCE log realistically provides."""

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, objective="reg:squarederror"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        self.model.load_model(str(path))


if nn is not None:

    class _LSTMNet(nn.Module):
        def __init__(self, input_size: int = 2, hidden_size: int = 32, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

else:
    _LSTMNet = None


class LSTMDemandForecaster(ForecastModel):
    """Primary model: captures temporal autocorrelation of surge/decay arrivals.

    Input features per timestep: [arrival_count, saturation_index].
    Sequence length: WINDOW_SIZE. Target: next-timestep arrival_count.
    """

    def __init__(self, hidden_size: int = 32, num_layers: int = 2, epochs: int = 50, lr: float = 1e-3):
        if torch is None:
            raise RuntimeError("torch is not installed; install it to use LSTMDemandForecaster")
        self.epochs = epochs
        self.lr = lr
        self.net = _LSTMNet(input_size=2, hidden_size=hidden_size, num_layers=num_layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # X shape: (n_samples, WINDOW_SIZE, 2); y shape: (n_samples,)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.net(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self.net(X_t).squeeze(-1).numpy()

    def save(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path) -> None:
        self.net.load_state_dict(torch.load(path))


def build_sliding_windows(series: np.ndarray, saturation: np.ndarray, window: int = WINDOW_SIZE):
    """Builds (X, y) supervised pairs from a raw arrivals/saturation time series."""
    X, y = [], []
    for i in range(len(series) - window):
        X.append(np.stack([series[i : i + window], saturation[i : i + window]], axis=-1))
        y.append(series[i + window])
    return np.array(X), np.array(y)
