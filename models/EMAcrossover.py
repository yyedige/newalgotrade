# ============================================
# EMA Trend + Price Confirmation Strategy
# SQLite IN, CSV OUT (LONG ONLY)
# ============================================

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

EMA_FAST = config.EMA_FAST
EMA_SLOW = config.EMA_SLOW
TICKER = config.TICKER

# ====================
# LOAD DATA FROM SQLITE
# ====================
db_path = BASE_DIR / "data" / "processed" / "data_processed.sqlite"

with sqlite3.connect(db_path) as conn:
    df = pd.read_sql(
        "SELECT * FROM data ORDER BY Date",
        conn,
        parse_dates=["Date"]
    )

df.set_index("Date", inplace=True)

print(f"Loaded {len(df)} rows from SQLite")

# ====================
# VALIDATION
# ====================
required_cols = ["Close", f"EMA{EMA_FAST}", f"EMA{EMA_SLOW}"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' missing")

# ====================
# STRATEGY LOGIC
# Long-only EMA trend + price filter
# ====================
df["signalEMA"] = 0

for i in range(1, len(df)):
    price = df["Close"].iloc[i]
    ema_fast = df[f"EMA{EMA_FAST}"].iloc[i]
    ema_slow = df[f"EMA{EMA_SLOW}"].iloc[i]

    # BUY / HOLD
    if (
        ema_fast > ema_slow and
        price >= ema_fast and
        price >= ema_slow
    ):
        df.iloc[i, df.columns.get_loc("signalEMA")] = 1
    else:
        df.iloc[i, df.columns.get_loc("signalEMA")] = 0

# ====================
# BACKTESTING
# ====================
df["market_ret"] = df["Close"].pct_change().fillna(0)
df["strategy_ret"] = df["signalEMA"].shift(1).fillna(0) * df["market_ret"]
df["strategy_eq"] = (1 + df["strategy_ret"]).cumprod()

# ====================
# METRICS
# ====================
long_days = (df["signalEMA"] == 1).sum()
flat_days = (df["signalEMA"] == 0).sum()

total_return = (df["strategy_eq"].iloc[-1] - 1) * 100
market_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

sharpe = (
    df["strategy_ret"].mean()
    / df["strategy_ret"].std()
    * np.sqrt(252)
    if df["strategy_ret"].std() > 0
    else 0
)

print("\n" + "=" * 60)
print("EMA TREND + PRICE CONFIRMATION STRATEGY (LONG ONLY)")
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
print("\nLast 10 signals:")
print(df[[
    "Close",
    f"EMA{EMA_FAST}",
    f"EMA{EMA_SLOW}",
    "signalEMA"
]].tail(10))

position_changes = df[df["signalEMA"].diff() != 0]
print(f"\nTotal position changes: {len(position_changes)}")

# ====================
# SAVE SIGNALS TO CSV
# ====================
out_path = BASE_DIR / "data" / "signals" / "ema_signal.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

signal_df = df.reset_index()[[
    "Date",
    "Close",
    f"EMA{EMA_FAST}",
    f"EMA{EMA_SLOW}",
    "signalEMA",
    "strategy_ret"
]]

signal_df.to_csv(out_path, index=False)
print(f"\nSaved EMA signals CSV to: {out_path}")