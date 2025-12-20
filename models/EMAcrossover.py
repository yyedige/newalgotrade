# ============================================
# EMA Crossover Strategy - SQLite IN, CSV OUT
# ============================================

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
    df = pd.read_sql("SELECT * FROM data ORDER BY Date", conn, parse_dates=["Date"])

df.set_index("Date", inplace=True)

print(f"Loaded {len(df)} rows from SQLite")

# Check columns exist
if f"EMA{EMA_FAST}" not in df.columns or f"EMA{EMA_SLOW}" not in df.columns:
    raise KeyError(f"EMA columns missing. Available: {df.columns.tolist()}")

# Generate signals
df["signalEMA"] = np.where(df[f"EMA{EMA_FAST}"] > df[f"EMA{EMA_SLOW}"], 1, -1)

# Calculate returns
df["market_ret"] = df["Close"].pct_change().fillna(0)
df["strategy_ret"] = df["signalEMA"].shift(1).fillna(0) * df["market_ret"]
df["strategy_eq"] = (1 + df["strategy_ret"]).cumprod()

# Metrics
total_return = (df["strategy_eq"].iloc[-1] - 1) * 100
market_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
sharpe = df["strategy_ret"].mean() / df["strategy_ret"].std() * np.sqrt(252) if df["strategy_ret"].std() > 0 else 0

print("\n" + "="*60)
print("EMA CROSSOVER STRATEGY")
print("="*60)
print(f"Strategy Return: {total_return:.2f}%")
print(f"Market Return: {market_return:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print("="*60)

print("\nLast 10 signals:")
print(df[["Close", f"EMA{EMA_FAST}", f"EMA{EMA_SLOW}", "signalEMA"]].tail(10))

# Save ONLY signals to CSV
out_path = BASE_DIR / "data" / "signals" / "ema_signal.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

signal_df = df.reset_index()[["Date", "Close", f"EMA{EMA_FAST}", f"EMA{EMA_SLOW}", "signalEMA", "strategy_ret"]]
signal_df.to_csv(out_path, index=False)

print(f"\nSaved signals CSV to: {out_path}")