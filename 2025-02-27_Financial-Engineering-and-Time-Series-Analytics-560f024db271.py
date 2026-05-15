# Description: Short example for Financial Engineering and Time Series Analytics.


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, hidden: int = 50, output_size: int = 1,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 15) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


    df = pd.DataFrame({"price": [100, 102, 105, 107, 110, 115]})
    df["lag_1"] = df["price"].shift(1)
    df["diff"] = df["price"].diff()
    logger.info(df)

    df["returns"] = df["price"].pct_change()
    df["volatility"] = df["returns"].rolling(window=3).std()
    logger.info(df)

    model = ARIMA(df["price"], order=(1, 1, 1))
    model_fit = model.fit()
    logger.info(model_fit.summary())


    X_train = np.random.rand(100, 10, 1)  # Simulated time series data
    y_train = np.random.rand(100, 1)
    model = Sequential(
        [LSTM(50, return_sequences=True, input_shape=(10, 1)), LSTM(50), Dense(1)]
    )
        _train_torch(model, X_train, y_train)


if __name__ == "__main__":
    main()
