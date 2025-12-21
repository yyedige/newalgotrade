from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

# ====================
# PATHS
# ====================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# ====================
# LOAD CONFIG
# ====================
import importlib.util
spec = importlib.util.spec_from_file_location("config", BASE_DIR / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

SMA_FAST = config.EMA_FAST   # SMA50
SMA_SLOW = config.EMA_SLOW   # SMA200
TICKER = config.TICKER

# ====================
# LOAD DATA FROM SQLITE
# ====================
db_path = BASE_DIR / "data" / "processed" / "data_processed.sqlite"

with sqlite3.connect(db_path) as conn:
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )
    if tables.empty:
        raise RuntimeError(f"No tables found in {db_path}")

    table_name = tables["name"].iloc[0]
    df = pd.read_sql(
        f'SELECT * FROM "{table_name}"',
        conn,
        parse_dates=["Date"]
    )

df = df.sort_values("Date").set_index("Date")

print(f"Loaded {len(df)} rows from SQLite")

# ====================
# VALIDATION
# ====================
required_cols = ["Close", f"SMA{SMA_FAST}", f"SMA{SMA_SLOW}"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found")

# ====================
# STRATEGY LOGIC
# Long-only trend + price confirmation
# ====================
df["position"] = 0

for i in range(1, len(df)):
    price = df["Close"].iloc[i]
    sma_fast = df[f"SMA{SMA_FAST}"].iloc[i]
    sma_slow = df[f"SMA{SMA_SLOW}"].iloc[i]

    # BUY / HOLD
    if (
        sma_fast > sma_slow and
        price >= sma_fast and
        price >= sma_slow
    ):
        df.iloc[i, df.columns.get_loc("position")] = 1
    else:
        df.iloc[i, df.columns.get_loc("position")] = 0

df["signalSMA"] = df["position"]

# ====================
# BACKTESTING
# ====================
df["market_ret"] = df["Close"].pct_change().fillna(0)
df["strategy_ret"] = df["signalSMA"].shift(1).fillna(0) * df["market_ret"]

df["strategy_eq"] = (1 + df["strategy_ret"]).cumprod()
df["market_eq"] = (1 + df["market_ret"]).cumprod()

# ====================
# METRICS
# ====================
long_days = (df["signalSMA"] == 1).sum()
flat_days = (df["signalSMA"] == 0).sum()

total_return = (df["strategy_eq"].iloc[-1] - 1) * 100
market_return = (df["market_eq"].iloc[-1] - 1) * 100

sharpe = (
    df["strategy_ret"].mean()
    / df["strategy_ret"].std()
    * np.sqrt(252)
    if df["strategy_ret"].std() > 0
    else 0
)

print("\n" + "=" * 60)
print("SMA TREND + PRICE CONFIRMATION STRATEGY (LONG ONLY)")
print("=" * 60)
print(f"Strategy Return: {total_return:.2f}%")
print(f"Market Return:   {market_return:.2f}%")
print(f"Sharpe Ratio:    {sharpe:.2f}")
print("\nPosition Days:")
print(f"  Long: {long_days} days ({long_days / len(df) * 100:.1f}%)")
print(f"  Flat: {flat_days} days ({flat_days / len(df) * 100:.1f}%)")
print("=" * 60)

# ====================
# DEBUG CHECKS
# ====================
print("\nLast 10 rows:")
print(df[["Close", f"SMA{SMA_FAST}", f"SMA{SMA_SLOW}", "signalSMA"]].tail(10))

position_changes = df[df["signalSMA"].diff() != 0]
print(f"\nTotal position changes: {len(position_changes)}")
print(position_changes[
    ["Close", f"SMA{SMA_FAST}", f"SMA{SMA_SLOW}", "signalSMA"]
].head(5))

# ====================
# SAVE SIGNALS
# ====================
out_csv_path = BASE_DIR / "data" / "signals" / "sma_signal.csv"
out_csv_path.parent.mkdir(parents=True, exist_ok=True)

signal_df = df.reset_index()[[
    "Date", "Close", f"SMA{SMA_FAST}", f"SMA{SMA_SLOW}",
    "signalSMA", "strategy_ret"
]]

signal_df.to_csv(out_csv_path, index=False)
print(f"\nSaved SMA signals CSV to: {out_csv_path}")

print("\nBacktest complete (no plots generated).")