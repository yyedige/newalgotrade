# models/lstm.py
# ============================================
# LSTM DIRECTION CLASSIFIER ON PIPELINE DATA
# ============================================

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "data" / "processed" / "data_processed.sqlite"

SEQ_LEN = 21
TRAIN_SPLIT = 0.8
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 66

np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Data loading utilities
# ----------------------------

def load_from_sqlite(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_name = tables["name"].iloc[0]
        df = pd.read_sql(
            f'SELECT Date, Close, Volume FROM "{table_name}" ORDER BY Date',
            conn,
            parse_dates=["Date"],
        )
    df = df.set_index("Date").sort_index()
    return df

def compute_returns(df, price_col="Close"):
    px = df[price_col].astype(float)
    rets = np.log(px / px.shift(1))
    df = df.copy()
    df["return"] = rets
    df.dropna(inplace=True)
    return df

def to_binary(y):
    return (y > 0).astype(int)

def scale_features(df, feature_cols):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    return df_scaled, scaler

def build_sequences(df, feature_cols, label_col="return", seq_len=SEQ_LEN):
    features = df[feature_cols].values
    labels = to_binary(df[label_col].values)
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(labels[i+seq_len])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def train_val_split(X, y, train_split=TRAIN_SPLIT):
    n = len(X)
    n_train = int(n * train_split)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    return X_train, y_train, X_val, y_val

# ----------------------------
# Dataset & Model
# ----------------------------

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=100, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.out(self.fc2(out))
        return out.squeeze(1)

# ----------------------------
# Baseline & training
# ----------------------------

def baseline_direction_accuracy(y_true):
    n = len(y_true)
    accuracies = []
    for _ in range(1000):
        y_rand = np.random.randint(0, 2, size=n)
        acc = accuracy_score(y_true, y_rand)
        accuracies.append(acc)
    return float(np.mean(accuracies))

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.float().to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_val_preds, all_val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.float().to(DEVICE)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

                all_val_preds.extend((y_pred.cpu().numpy() > 0.5).astype(int))
                all_val_true.extend(y_batch.cpu().numpy().astype(int))

        val_acc = accuracy_score(all_val_true, all_val_preds)
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {np.mean(train_losses):.4f} | "
            f"Val loss: {np.mean(val_losses):.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

    return model

# ----------------------------
# Robust evaluation
# ----------------------------

def evaluate_model(model, X_test, y_test, n_preds_mc=1):
    model.eval()
    X_test_t = torch.from_numpy(X_test).to(DEVICE)
    all_preds = []
    with torch.no_grad():
        for _ in range(n_preds_mc):
            y_prob = model(X_test_t).cpu().numpy()
            y_hat = (y_prob > 0.5).astype(int)
            all_preds.append(y_hat)

    votes = np.mean(np.stack(all_preds, axis=0), axis=0)
    y_pred = (votes > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if np.all(y_test == 0):
            tn = cm[0, 0]
        elif np.all(y_test == 1):
            tp = cm[0, 0]

    print(f"Test accuracy: {acc:.4f}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    return acc, cm, y_pred

# ----------------------------
# Main entry point
# ----------------------------

def main():
    # 1. Load and prepare data
    df = load_from_sqlite(DB_PATH)
    df = compute_returns(df, price_col="Close")

    feature_cols = ["Close", "Volume", "return"]
    df = df.dropna(subset=feature_cols)

    df_scaled, scaler = scale_features(df, feature_cols)

    X, y = build_sequences(df_scaled, feature_cols, label_col="return", seq_len=SEQ_LEN)

    X_train, y_train, X_val, y_val = train_val_split(X, y, TRAIN_SPLIT)
    n_val = len(X_val)
    n_test = n_val // 2
    X_test, y_test = X_val[:n_test], y_val[:n_test]
    X_val, y_val = X_val[n_test:], y_val[n_test:]

    base_acc = baseline_direction_accuracy(y_test)
    print(f"Random baseline accuracy: {base_acc:.4f}")

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = X.shape[2]
    model = LSTMClassifier(n_features=n_features, hidden_size=100, num_layers=2, dropout=0.2)

    # 2. Train and evaluate
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    evaluate_model(model, X_test, y_test, n_preds_mc=10)

    # 3. Build full-series signals
    X_all, y_all = build_sequences(df_scaled, feature_cols, label_col="return", seq_len=SEQ_LEN)
    dates_all = df.index[SEQ_LEN:]

    model.eval()
    X_all_t = torch.from_numpy(X_all).to(DEVICE)
    with torch.no_grad():
        y_prob_all = model(X_all_t).cpu().numpy()
    y_hat_all = (y_prob_all > 0.5).astype(int)
    signals = np.where(y_hat_all == 1, 1, -1)

    signal_df = pd.DataFrame(
        {
            "Date": dates_all,
            "Close": df["Close"].iloc[SEQ_LEN:].values,
            "signalLSTM": signals,
        }
    )

    signals_dir = BASE_DIR / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    out_signal_path = signals_dir / "lstm_direction_signal.csv"
    signal_df.to_csv(out_signal_path, index=False)
    print(f"LSTM direction signals saved to: {out_signal_path}")

    # 4. Backtest plot for this model
    import matplotlib.pyplot as plt

    signal_df = signal_df.sort_values("Date").reset_index(drop=True)
    signal_df["market_ret"] = signal_df["Close"].pct_change().fillna(0)
    signal_df["strategy_ret"] = signal_df["signalLSTM"].shift(1).fillna(0) * signal_df["market_ret"]
    signal_df["eq_market"] = (1 + signal_df["market_ret"]).cumprod()
    signal_df["eq_lstm"] = (1 + signal_df["strategy_ret"]).cumprod()

    results_dir = BASE_DIR / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "lstm_direction_backtest.png"

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(signal_df["Date"], signal_df["eq_market"], label="Buy & Hold", color="black", linewidth=1.5)
    ax.plot(signal_df["Date"], signal_df["eq_lstm"], label="LSTM Direction", color="tab:blue", linewidth=1.5)
    ax.set_title("LSTM Direction Strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalised)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"LSTM backtest plot saved to: {plot_path}")

    # 5. Save model weights
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "lstm_stock_direction.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    main()