# ============================================
# Ensemble Signal - Fixed for Pipeline
# ============================================

from pathlib import Path
import pandas as pd
import numpy as np
import os

# Get project root
BASE_DIR = Path(os.getcwd())
if BASE_DIR.name == "models":
    BASE_DIR = BASE_DIR.parent

print(f"Project root: {BASE_DIR}")

signals_dir = BASE_DIR / "data" / "signals"

# Load all signal CSVs
ema_df = pd.read_csv(signals_dir / "ema_signal.csv", parse_dates=["Date"])
sma_df = pd.read_csv(signals_dir / "sma_signal.csv", parse_dates=["Date"])
arima_df = pd.read_csv(signals_dir / "arima_garch_signal.csv", parse_dates=["Date"])
lstm_df = pd.read_csv(signals_dir / "bilstm_diff_signal.csv", parse_dates=["Date"])

# Merge on Date
ensemble = ema_df[["Date", "Close"]].copy()
ensemble = ensemble.merge(ema_df[["Date", "signalEMA"]], on="Date", how="outer")
ensemble = ensemble.merge(sma_df[["Date", "signalSMA"]], on="Date", how="outer")
ensemble = ensemble.merge(arima_df[["Date", "signal"]].rename(columns={"signal": "signalARIMA"}), on="Date", how="outer")

# Check what column name is in lstm_df
print("LSTM columns:", lstm_df.columns.tolist())

# Use the correct signal column from bilstm_diff_signal.csv
if "signal" in lstm_df.columns:
    ensemble = ensemble.merge(lstm_df[["Date", "signal"]].rename(columns={"signal": "signalLSTM"}), on="Date", how="outer")
elif "pred_diff" in lstm_df.columns:
    # Convert pred_diff to signal
    lstm_df["signalLSTM"] = np.where(lstm_df["pred_diff"] > 0, 1, -1)
    ensemble = ensemble.merge(lstm_df[["Date", "signalLSTM"]], on="Date", how="outer")

ensemble = ensemble.sort_values("Date").fillna(0).reset_index(drop=True)

# SIMPLE MAJORITY VOTE
ensemble["vote_sum"] = (
    ensemble["signalEMA"] + 
    ensemble["signalSMA"] + 
    ensemble["signalARIMA"] + 
    ensemble["signalLSTM"]
)

ensemble["final_signal"] = np.sign(ensemble["vote_sum"])

print("\nSignal distribution:")
print(ensemble["final_signal"].value_counts())

# ====================
# BACKTEST ALL STRATEGIES
# ====================

ensemble["market_ret"] = ensemble["Close"].pct_change().fillna(0)

# Strategy returns
ensemble["ret_ema"] = ensemble["signalEMA"].shift(1).fillna(0) * ensemble["market_ret"]
ensemble["ret_sma"] = ensemble["signalSMA"].shift(1).fillna(0) * ensemble["market_ret"]
ensemble["ret_arima"] = ensemble["signalARIMA"].shift(1).fillna(0) * ensemble["market_ret"]
ensemble["ret_lstm"] = ensemble["signalLSTM"].shift(1).fillna(0) * ensemble["market_ret"]
ensemble["ret_ensemble"] = ensemble["final_signal"].shift(1).fillna(0) * ensemble["market_ret"]

# Cumulative equity curves
ensemble["eq_market"] = (1 + ensemble["market_ret"]).cumprod()
ensemble["eq_ema"] = (1 + ensemble["ret_ema"]).cumprod()
ensemble["eq_sma"] = (1 + ensemble["ret_sma"]).cumprod()
ensemble["eq_arima"] = (1 + ensemble["ret_arima"]).cumprod()
ensemble["eq_lstm"] = (1 + ensemble["ret_lstm"]).cumprod()
ensemble["eq_ensemble"] = (1 + ensemble["ret_ensemble"]).cumprod()

# ====================
# PERFORMANCE METRICS
# ====================

def calculate_metrics(returns, name):
    total_return = (returns + 1).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    wins = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        "Strategy": name,
        "Total Return (%)": total_return * 100,
        "Annual Return (%)": ann_return * 100,
        "Annual Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd * 100,
        "Win Rate (%)": win_rate,
        "Total Trades": total_trades
    }

metrics = []
metrics.append(calculate_metrics(ensemble["market_ret"], "Buy & Hold"))
metrics.append(calculate_metrics(ensemble["ret_ema"], "EMA Crossover"))
metrics.append(calculate_metrics(ensemble["ret_sma"], "SMA Crossover"))
metrics.append(calculate_metrics(ensemble["ret_arima"], "ARIMA+GARCH"))
metrics.append(calculate_metrics(ensemble["ret_lstm"], "BiLSTM"))
metrics.append(calculate_metrics(ensemble["ret_ensemble"], "ENSEMBLE (Vote)"))

results_df = pd.DataFrame(metrics)

# ====================
# PRINT RESULTS
# ====================

print("\n" + "="*80)
print("STRATEGY COMPARISON - BACKTESTING RESULTS")
print("="*80)
print(f"Period: {ensemble['Date'].iloc[0].date()} to {ensemble['Date'].iloc[-1].date()}")
print(f"Total Days: {len(ensemble)}")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

best_sharpe = results_df.loc[results_df["Sharpe Ratio"].idxmax()]
best_return = results_df.loc[results_df["Total Return (%)"].idxmax()]

print(f"\n*** Best Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")
print(f"*** Best Total Return: {best_return['Strategy']} ({best_return['Total Return (%)']:.2f}%)")

# ====================
# SAVE RESULTS
# ====================

out_signal_path = signals_dir / "ensemble_simple_vote.csv"
ensemble.to_csv(out_signal_path, index=False)
print(f"\nSaved ensemble signals to: {out_signal_path}")

out_results_path = BASE_DIR / "data" / "results" / "strategy_comparison.csv"
out_results_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(out_results_path, index=False)
print(f"Saved performance metrics to: {out_results_path}")

# ====================
# NO PLOTS (automated pipeline)
# ====================

print("\nBacktesting complete!")