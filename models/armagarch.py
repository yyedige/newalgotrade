# ============================================
# ARIMA+GARCH Strategy - JUPYTER VERSION
# ============================================

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# Get project root - JUPYTER VERSION
BASE_DIR = Path(os.getcwd())
if BASE_DIR.name == "models":
    BASE_DIR = BASE_DIR.parent

print(f"Project root: {BASE_DIR}")

# Load config
import importlib.util
spec = importlib.util.spec_from_file_location("config", BASE_DIR / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Configuration
DB_PATH = BASE_DIR / "data" / "processed" / "data_processed.sqlite"
START_DATE = config.BACKTEST_START_DATE
WINDOW_LENGTH = config.WINDOW_LENGTH
TICKER = config.TICKER
P_MAX = 3
Q_MAX = 3
REBALANCE_DAYS = 20

def load_data_from_sqlite():
    """Load stock data from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if tables.empty:
            raise RuntimeError(f"No tables found in {DB_PATH}")
        table_name = tables["name"].iloc[0]
        
        df = pd.read_sql(
            f'SELECT Date, Close FROM "{table_name}" ORDER BY Date', 
            conn, 
            parse_dates=["Date"]
        )
    
    df = df.set_index("Date")
    df['returns'] = np.log(df['Close']).diff()
    df = df.dropna(subset=['returns'])
    
    return df

# Load data
df = load_data_from_sqlite()
returns = df['returns'].values
dates = df.index

print(f"Total data: {len(df)} days ({dates[0].date()} to {dates[-1].date()})")

# Find START_DATE index
start_idx = df.index.get_indexer([pd.to_datetime(START_DATE)], method='nearest')[0]
print(f"Trading starts: {dates[start_idx].date()} (index {start_idx})")

if start_idx < WINDOW_LENGTH:
    raise ValueError(f"Need {WINDOW_LENGTH} days before {START_DATE}. Only have {start_idx} days.")

# Calculate forecasts
forecasts_length = len(returns) - start_idx
forecasts = np.zeros(forecasts_length)
directions = np.zeros(forecasts_length)

# Rebalance points
rebalance_points = list(range(0, forecasts_length, REBALANCE_DAYS))
if rebalance_points[-1] != forecasts_length - 1:
    rebalance_points.append(forecasts_length - 1)

print(f"\nProcessing {len(rebalance_points)} rebalance points...\n")

current_forecast = 0
current_direction = 1

for idx, i in enumerate(rebalance_points):
    print(f"Rebalance {idx+1}/{len(rebalance_points)} - Day {i}/{forecasts_length} ({100*i/forecasts_length:.1f}%)")
    
    window_start = start_idx + i - WINDOW_LENGTH
    window_end = start_idx + i
    roll_returns = returns[window_start:window_end]
    
    # Find best ARIMA
    final_aic = np.inf
    final_order = (1, 0, 1)
    
    for p in range(1, P_MAX + 1):
        for q in range(1, Q_MAX + 1):
            try:
                model = ARIMA(roll_returns, order=(p, 0, q)).fit(disp=0)
                if model.aic < final_aic:
                    final_aic = model.aic
                    final_order = (p, 0, q)
            except:
                continue
    
    # Fit GARCH
    try:
        spec = arch_model(
            roll_returns,
            mean='ARX',
            lags=final_order[0],
            vol='GARCH',
            p=1,
            q=1,
            dist='skewt'
        )
        fit = spec.fit(disp='off', show_warning=False)
        
        fore = fit.forecast(horizon=1)
        current_forecast = fore.mean.iloc[-1, 0]
        current_direction = 1 if current_forecast > 0 else -1
        
    except:
        current_forecast = 0
        current_direction = 1
    
    # Fill forecast until next rebalance
    if idx < len(rebalance_points) - 1:
        next_i = rebalance_points[idx + 1]
        forecasts[i:next_i] = current_forecast
        directions[i:next_i] = current_direction
    else:
        forecasts[i:] = current_forecast
        directions[i:] = current_direction

print("\nForecasting complete!\n")

# Create signals dataframe
forecast_dates = dates[start_idx:]
signals_df = pd.DataFrame({
    'Date': forecast_dates,
    'Close': df['Close'].iloc[start_idx:].values,
    'forecast_return': forecasts,
    'signal': directions
})

# Calculate strategy performance
signals_df['market_ret'] = returns[start_idx:]
signals_df['strategy_ret'] = signals_df['signal'].shift(1).fillna(0) * signals_df['market_ret']
signals_df['strategy_eq'] = (1 + signals_df['strategy_ret']).cumprod()
signals_df['market_eq'] = (1 + signals_df['market_ret']).cumprod()

# Performance metrics
strategy_returns = signals_df['strategy_ret']
market_returns = signals_df['market_ret']

total_strategy = (signals_df['strategy_eq'].iloc[-1] - 1) * 100
total_market = (signals_df['market_eq'].iloc[-1] - 1) * 100

n_trades = (signals_df['signal'].diff() != 0).sum()
n_long = (signals_df['signal'] == 1).sum()
n_short = (signals_df['signal'] == -1).sum()
n_wins = (strategy_returns > 0).sum()
win_rate = (n_wins / len(strategy_returns[strategy_returns != 0]) * 100) if len(strategy_returns[strategy_returns != 0]) > 0 else 0

sharpe_strategy = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
sharpe_market = (market_returns.mean() / market_returns.std() * np.sqrt(252)) if market_returns.std() > 0 else 0

cum_strategy = (1 + strategy_returns).cumprod()
running_max = cum_strategy.expanding().max()
drawdown = (cum_strategy - running_max) / running_max
max_dd = drawdown.min() * 100

print("="*60)
print("ARIMA+GARCH PERFORMANCE")
print("="*60)
print(f"Period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
print(f"Rebalancing: Every {REBALANCE_DAYS} days")
print(f"\nStrategy Return: {total_strategy:>12.2f}%")
print(f"Buy & Hold Return: {total_market:>12.2f}%")
print(f"Excess Return: {total_strategy - total_market:>12.2f}%")
print(f"\nStrategy Sharpe: {sharpe_strategy:>12.2f}")
print(f"Buy & Hold Sharpe: {sharpe_market:>12.2f}")
print(f"\nMax Drawdown: {max_dd:>12.2f}%")
print(f"Win Rate: {win_rate:>12.2f}%")
print(f"\nTotal Trades: {n_trades:>12.0f}")
print(f"Long Days: {n_long:>12.0f}")
print(f"Short Days: {n_short:>12.0f}")
print("="*60)

# Save signals to CSV
out_path = BASE_DIR / "data" / "signals" / "arima_garch_signal.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

signals_df.to_csv(out_path, index=False)
print(f"\nSaved signals CSV to: {out_path}")

# Plot
plt.figure(figsize=(14, 7))

strategy_curve = strategy_returns.cumsum()
market_curve = market_returns.cumsum()

plt.plot(forecast_dates, strategy_curve.values, color='green', 
         label=f'ARIMA+GARCH Strategy (rebal every {REBALANCE_DAYS}d)', linewidth=2)
plt.plot(forecast_dates, market_curve.values, color='red', 
         label='Buy & Hold', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Log Return', fontsize=12)
plt.title(f'{TICKER} - ARIMA+GARCH Strategy vs Buy & Hold (from {START_DATE})', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()