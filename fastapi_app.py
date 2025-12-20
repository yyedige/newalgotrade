from pathlib import Path
from datetime import datetime
import asyncio
import io
import base64

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
import papermill as pm

app = FastAPI(title="NEWALGOTRADE Pipeline")

PIPELINE_STATUS = {
    "last_run_start": None,
    "last_run_end": None,
    "last_run_ok": None,
    "last_error": None,
    "current_step": None,
    "log": [],
    "is_running": False,
}

# Project root
project_root = Path(__file__).resolve().parent

# Notebooks to execute
NOTEBOOK_PATHS = [
    project_root / "data_load.ipynb",
    project_root / "data_processer.ipynb",
    project_root / "EMA50EMA200.ipynb",
    project_root / "SMA50SMA200.ipynb",
    project_root / "signals_risk.ipynb",
    project_root / "backtersers.ipynb",
]

executed_dir = project_root / "executed"
executed_dir.mkdir(exist_ok=True)
print(f"Executed notebooks will be saved to: {executed_dir}")

# Results directory for plots
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)


def _log(msg: str):
    now = datetime.utcnow().isoformat()
    line = f"[{now}] {msg}"
    print(line)
    PIPELINE_STATUS["log"].append(line)


def run_pipeline():
    if PIPELINE_STATUS.get("is_running"):
        _log("Previous run still in progress, skipping.")
        return

    PIPELINE_STATUS["is_running"] = True
    start_time = datetime.utcnow()
    
    PIPELINE_STATUS["last_run_start"] = start_time.isoformat()
    PIPELINE_STATUS["last_run_end"] = None
    PIPELINE_STATUS["last_run_ok"] = None
    PIPELINE_STATUS["last_error"] = None
    PIPELINE_STATUS["current_step"] = "starting"
    
    _log("=== Pipeline run started ===")

    try:
        for nb_path in NOTEBOOK_PATHS:
            if not nb_path.exists():
                raise FileNotFoundError(f"Notebook not found: {nb_path}")

            PIPELINE_STATUS["current_step"] = f"running {nb_path.name}"
            _log(f"Executing: {nb_path.name}")

            output_nb = executed_dir / f"{nb_path.stem}_executed.ipynb"

            pm.execute_notebook(
                input_path=str(nb_path),
                output_path=str(output_nb),
                progress_bar=False,
                report_mode=False,
            )

            _log(f"✓ Finished: {nb_path.name}")

        PIPELINE_STATUS["current_step"] = "idle"
        PIPELINE_STATUS["last_run_ok"] = True
        _log("=== Pipeline finished successfully ===")

    except Exception as e:
        PIPELINE_STATUS["current_step"] = "error"
        PIPELINE_STATUS["last_run_ok"] = False
        PIPELINE_STATUS["last_error"] = str(e)
        _log(f"✗ Pipeline error: {e}")

    finally:
        PIPELINE_STATUS["last_run_end"] = datetime.utcnow().isoformat()
        PIPELINE_STATUS["is_running"] = False


async def periodic_runner():
    while True:
        try:
            run_pipeline()
        except Exception as e:
            _log(f"Unexpected error in periodic_runner: {e}")
        await asyncio.sleep(60)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log("FastAPI started. Periodic runner enabled (every 60 seconds).")
    task = asyncio.create_task(periodic_runner())
    yield
    task.cancel()

app = FastAPI(title="NEWALGOTRADE Pipeline", lifespan=lifespan)


@app.post("/run-pipeline")
async def run_pipeline_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline)
    return {"message": "Pipeline started in background"}


@app.get("/status")
async def status():
    return PIPELINE_STATUS


@app.get("/results", response_class=HTMLResponse)
async def results():
    """Display backtest results with chart and metrics."""
    import sqlite3
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # non-GUI backend
    import matplotlib.pyplot as plt
    
    try:
        # Load backtest data from combined.sqlite
        signals_db = project_root / "data" / "signals" / "combined.sqlite"
        price_db = project_root / "data" / "processed" / "data_processed.sqlite"
        
        if not signals_db.exists() or not price_db.exists():
            return "<h1>No results yet. Run the pipeline first.</h1>"
        
        # Load prices
        with sqlite3.connect(price_db) as conn:
            price_df = pd.read_sql(
                'SELECT Date, Close, SMA50, SMA200 FROM data',
                conn,
                parse_dates=["Date"]
            )
        price_df = price_df.set_index("Date").sort_index()
        
        # Load signals
        with sqlite3.connect(signals_db) as conn:
            sig_df = pd.read_sql(
                "SELECT Date, signal_combined FROM signals_combined",
                conn,
                parse_dates=["Date"]
            )
        sig_df = sig_df.set_index("Date").sort_index()
        
        # Merge
        df = price_df.join(sig_df, how="inner")
        df["signal_combined"] = df["signal_combined"].fillna(0).astype(int)
        
        # Filter from 2022-01-01
        df = df.loc[df.index >= pd.Timestamp("2022-01-01")].dropna(subset=["Close"])
        
        # Vectorized backtest
        initial_capital = 100_000.0
        df["ret"] = df["Close"].pct_change().fillna(0)
        position = df["signal_combined"].shift(1).fillna(0).replace(-1, 0)
        df["ret_signal"] = position * df["ret"]
        df["eq_signal"] = (1 + df["ret_signal"]).cumprod()
        df["eq_buyhold"] = (1 + df["ret"]).cumprod()
        df["portfolio_signal"] = initial_capital * df["eq_signal"]
        df["portfolio_buyhold"] = initial_capital * df["eq_buyhold"]
        
        # Metrics
        final_signal = df["portfolio_signal"].iloc[-1]
        final_bh = df["portfolio_buyhold"].iloc[-1]
        ret_signal = (final_signal - initial_capital) / initial_capital
        ret_bh = (final_bh - initial_capital) / initial_capital
        
        sharpe_signal = (
            df["ret_signal"].mean() / df["ret_signal"].std() * np.sqrt(252)
            if df["ret_signal"].std() != 0 else 0
        )
        sharpe_bh = (
            df["ret"].mean() / df["ret"].std() * np.sqrt(252)
            if df["ret"].std() != 0 else 0
        )
        
        def max_drawdown(values):
            running_max = values.cummax()
            dd = (values - running_max) / running_max
            return dd.min()
        
        mdd_signal = max_drawdown(df["portfolio_signal"])
        mdd_bh = max_drawdown(df["portfolio_buyhold"])
        
        # Plot
        fig, ax_price = plt.subplots(figsize=(13, 7))
        
        ax_price.plot(df.index, df["Close"], label="Close", color="black", linewidth=1.2, alpha=0.7)
        ax_price.plot(df.index, df["SMA50"], label="SMA 50", color="blue", linewidth=1.0)
        ax_price.plot(df.index, df["SMA200"], label="SMA 200", color="orange", linewidth=1.0)
        
        # Buy/sell arrows
        trade_signal = df["signal_combined"].shift(1).fillna(0)
        prev_pos = trade_signal.shift(1).fillna(0)
        buy_mask = (prev_pos <= 0) & (trade_signal > 0)
        sell_mask = (prev_pos > 0) & (trade_signal <= 0)
        buys = df[buy_mask]
        sells = df[sell_mask]
        
        ax_price.scatter(buys.index, buys["Close"], marker="^", color="green", s=80, label="Buy")
        ax_price.scatter(sells.index, sells["Close"], marker="v", color="red", s=80, label="Sell")
        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price")
        ax_price.grid(True)
        
        # Equity on secondary axis
        ax_eq = ax_price.twinx()
        ax_eq.plot(df.index, df["portfolio_buyhold"], label="Buy & Hold", color="purple", linewidth=1.4)
        ax_eq.plot(df.index, df["portfolio_signal"], label="Signal Strategy", color="green", linewidth=1.4, linestyle="--")
        ax_eq.set_ylabel("Portfolio Value ($)")
        ax_eq.axhline(initial_capital, color="gray", linestyle="--", linewidth=1)
        
        lines1, labels1 = ax_price.get_legend_handles_labels()
        lines2, labels2 = ax_eq.get_legend_handles_labels()
        ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        plt.title("Backtest Results (from 2022-01-01)")
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # HTML response
        html_content = f"""
        <html>
        <head><title>Backtest Results</title></head>
        <body style="font-family: Arial; margin: 20px;">
            <h1>Backtest Results</h1>
            <h2>Performance Metrics (from 2022-01-01)</h2>
            <table border="1" cellpadding="8" style="border-collapse: collapse;">
                <tr><th>Metric</th><th>Signal Strategy</th><th>Buy & Hold</th></tr>
                <tr><td>Final Value</td><td>${final_signal:,.2f}</td><td>${final_bh:,.2f}</td></tr>
                <tr><td>Total Return</td><td>{ret_signal:.2%}</td><td>{ret_bh:.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{sharpe_signal:.2f}</td><td>{sharpe_bh:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{mdd_signal:.2%}</td><td>{mdd_bh:.2%}</td></tr>
            </table>
            <h2>Chart</h2>
            <img src="data:image/png;base64,{img_base64}" style="max-width:100%;">
        </body>
        </html>
        """
        return html_content
        
    except Exception as e:
        return f"<h1>Error generating results</h1><pre>{e}</pre>"


@app.get("/")
async def root():
    return {
        "message": "NEWALGOTRADE Pipeline API",
        "endpoints": {
            "docs": "/docs",
            "status": "/status",
            "results": "/results",
            "run_pipeline": "/run-pipeline"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )