from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# Load config
import importlib.util
spec = importlib.util.spec_from_file_location("config", BASE_DIR / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

EMA_FAST = config.EMA_FAST
EMA_SLOW = config.EMA_SLOW
TICKER = config.TICKER

# Load from SQLite
db_path = BASE_DIR / "data" / "processed" / "data_processed.sqlite"

with sqlite3.connect(db_path) as conn:
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    if tables.empty:
        raise RuntimeError(f"No tables found in {db_path}")
    table_name = tables["name"].iloc[0]
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn, parse_dates=["Date"])

df = df.sort_values("Date").set_index("Date")

print(f"Loaded {len(df)} rows from SQLite")

# Check columns exist
for col in ["Close", f"SMA{EMA_FAST}", f"SMA{EMA_SLOW}"]:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found; columns = {df.columns.tolist()}")

# SMA strategy: signal = 1 if SMA_FAST > SMA_SLOW, else -1
df["signalSMA"] = np.where(df[f"SMA{EMA_FAST}"] > df[f"SMA{EMA_SLOW}"], 1, -1)

# Returns based on signal
df["market_ret"] = df["Close"].pct_change().fillna(0)
df["strategy_ret"] = df["signalSMA"].shift(1).fillna(0) * df["market_ret"]
df["strategy_eq"] = (1 + df["strategy_ret"]).cumprod()

# Metrics
total_return = (df["strategy_eq"].iloc[-1] - 1) * 100
market_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
sharpe = df["strategy_ret"].mean() / df["strategy_ret"].std() * np.sqrt(252) if df["strategy_ret"].std() > 0 else 0

print("\n" + "="*60)
print("SMA CROSSOVER STRATEGY")
print("="*60)
print(f"Strategy Return: {total_return:.2f}%")
print(f"Market Return: {market_return:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print("="*60)

print("\nLast 10 signals:")
print(df[["Close", f"SMA{EMA_FAST}", f"SMA{EMA_SLOW}", "signalSMA"]].tail(10))

# Save signals to CSV ONLY
out_csv_path = BASE_DIR / "data" / "signals" / "sma_signal.csv"
out_csv_path.parent.mkdir(parents=True, exist_ok=True)

signal_df = df.reset_index()[["Date", "Close", f"SMA{EMA_FAST}", f"SMA{EMA_SLOW}", "signalSMA", "strategy_ret"]]
signal_df.to_csv(out_csv_path, index=False)

print(f"\nSaved SMA signals CSV to: {out_csv_path}")