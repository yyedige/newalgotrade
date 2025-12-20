import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

# Get project root (go up TWO levels from data/ folder to root)
BASE_DIR = Path(__file__).parent.parent

# Load config dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("config", BASE_DIR / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

EMA_FAST = config.EMA_FAST
EMA_SLOW = config.EMA_SLOW

csv_path = BASE_DIR / "data" / "raw" / "data.csv"
out_path = BASE_DIR / "data" / "processed" / "data_processed.sqlite"

if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found at {csv_path.resolve()}")

raw = pd.read_csv(csv_path, header=None)

header = raw.iloc[0].tolist()
data = raw.iloc[2:].reset_index(drop=True)
data.columns = header

if "Price" not in data.columns:
    raise KeyError(f"'Price' column not found; got {data.columns.tolist()}")

data = data.rename(columns={"Price": "Date"})
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

num_cols = ["Close", "High", "Low", "Open", "Volume"]
for c in num_cols:
    if c not in data.columns:
        raise KeyError(f"Column '{c}' not found")
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=num_cols).reset_index(drop=True)

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ewm = pd.Series(gain, index=series.index).ewm(span=window, adjust=False).mean()
    loss_ewm = pd.Series(loss, index=series.index).ewm(span=window, adjust=False).mean()
    rs = gain_ewm / loss_ewm
    return 100 - (100 / (1 + rs))

data[f"SMA{EMA_FAST}"] = data["Close"].rolling(window=EMA_FAST, min_periods=EMA_FAST).mean()
data[f"SMA{EMA_SLOW}"] = data["Close"].rolling(window=EMA_SLOW, min_periods=EMA_SLOW).mean()
data[f"EMA{EMA_FAST}"] = data["Close"].ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
data[f"EMA{EMA_SLOW}"] = data["Close"].ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()
data["RSI14"] = compute_rsi(data["Close"], window=14)

keep_cols = ["Date", "Close", "Open", "High", "Low", "Volume",
             f"SMA{EMA_FAST}", f"SMA{EMA_SLOW}", f"EMA{EMA_FAST}", f"EMA{EMA_SLOW}", "RSI14"]
data = data[keep_cols].dropna().reset_index(drop=True)

out_path.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(out_path) as conn:
    data.to_sql("data", conn, if_exists="replace", index=False)