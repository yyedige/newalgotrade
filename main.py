from pathlib import Path
from datetime import datetime
import asyncio
import sys
import logging
import traceback
import subprocess
from concurrent.futures import ThreadPoolExecutor
import base64

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ---------- basic setup ----------
PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("newalgotrade")

PIPELINE_STATUS = {
    "last_run_start": None,
    "last_run_end": None,
    "last_run_ok": None,
    "last_error": None,
    "current_step": None,
    "is_running": False,
    "log": [],
}

PIPELINE_STEPS = [
    ("Data load",        PROJECT_ROOT / "data" / "data_load.py"),
    ("Data processing",  PROJECT_ROOT / "data" / "data_processer.py"),
    ("EMA crossover",    PROJECT_ROOT / "models" / "EMAcrossover.py"),
    ("SMA crossover",    PROJECT_ROOT / "models" / "SMAcrossover.py"),
    ("ARIMA+GARCH",      PROJECT_ROOT / "models" / "armagarch.py"),
    ("BiLSTM signals",   PROJECT_ROOT / "models" / "lstm.py"),
    ("Ensemble signals", PROJECT_ROOT / "total_signal.py"),
]

def _log(msg: str):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    log.info(msg)
    PIPELINE_STATUS["log"].append(line)
    PIPELINE_STATUS["log"] = PIPELINE_STATUS["log"][-500:]

# ---------- sync subprocess runner ----------
def run_script_sync(script_path: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=600,
    )
    return result.returncode, result.stdout, result.stderr

# ---------- pipeline runner ----------
async def run_pipeline():
    if PIPELINE_STATUS["is_running"]:
        _log("Pipeline already running, skipping new run.")
        return

    PIPELINE_STATUS["is_running"] = True
    PIPELINE_STATUS["last_run_start"] = datetime.utcnow().isoformat()
    PIPELINE_STATUS["last_run_end"] = None
    PIPELINE_STATUS["last_error"] = None
    PIPELINE_STATUS["last_run_ok"] = None

    _log("=== PIPELINE START ===")

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        for name, script in PIPELINE_STEPS:
            if not script.exists():
                _log(f"⚠️ Script not found: {script}, skipping step.")
                continue

            if script.suffix == ".ipynb":
                _log(f"⚠️ Notebook step skipped (convert to .py first): {script}")
                continue

            PIPELINE_STATUS["current_step"] = name
            _log(f"Running step: {name} ({script.name})")

            try:
                returncode, stdout, stderr = await loop.run_in_executor(
                    executor, run_script_sync, script
                )

                if stdout:
                    out_lines = stdout.strip().splitlines()
                    if out_lines:
                        _log(f"Output: {out_lines[-1]}")

                if returncode != 0:
                    _log(f"❌ {name} failed with exit code {returncode}")
                    if stderr:
                        _log(f"Error: {stderr[:500]}")
                    continue

                _log(f"✅ Finished step: {name}")

            except subprocess.TimeoutExpired:
                _log(f"❌ {name} timed out (exceeded 10 minutes)")
                continue
            except Exception as step_error:
                _log(f"❌ {name} exception: {type(step_error).__name__}: {str(step_error)}")
                _log(f"Traceback: {traceback.format_exc()[:500]}")
                continue

        PIPELINE_STATUS["last_run_ok"] = True
        _log("=== PIPELINE FINISHED OK ===")

    except Exception as e:
        PIPELINE_STATUS["last_run_ok"] = False
        error_msg = f"{type(e).__name__}: {str(e)}"
        PIPELINE_STATUS["last_error"] = error_msg
        _log(f"PIPELINE ERROR: {error_msg}")
        _log(f"Traceback: {traceback.format_exc()[:500]}")

    finally:
        PIPELINE_STATUS["last_run_end"] = datetime.utcnow().isoformat()
        PIPELINE_STATUS["current_step"] = None
        PIPELINE_STATUS["is_running"] = False

# ---------- periodic runner ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _log("FastAPI startup")

    async def periodic_runner():
        while True:
            _log("⏰ Scheduled pipeline run")
            await run_pipeline()
            await asyncio.sleep(180)

    task = asyncio.create_task(periodic_runner())
    yield
    _log("FastAPI shutdown")
    task.cancel()

# ---------- FastAPI app ----------
app = FastAPI(title="newalgotrade API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "newalgotrade API",
        "status": PIPELINE_STATUS["current_step"] or "idle",
        "is_running": PIPELINE_STATUS["is_running"],
        "endpoints": {
            "status": "/status",
            "run": "/run-pipeline",
            "logs": "/logs",
            "dashboard": "/dashboard",
            "backtest_plot": "/backtest-plot",
            "lstm_plot": "/lstm-plot",
        },
    }

@app.get("/status")
async def status():
    return JSONResponse(PIPELINE_STATUS)

@app.get("/logs")
async def logs():
    return {"lines": PIPELINE_STATUS["log"][-200:]}

@app.post("/run-pipeline")
async def run_pipeline_endpoint(bg: BackgroundTasks):
    if PIPELINE_STATUS["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")
    bg.add_task(run_pipeline)
    return {"message": "Pipeline started"}

# ---------- direct image URLs ----------
@app.get("/backtest-plot")
async def backtest_plot():
    chart_path = PROJECT_ROOT / "data" / "results" / "ensemble_backtest.png"
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Backtest plot not found")
    return FileResponse(chart_path)

@app.get("/lstm-plot")
async def lstm_plot():
    chart_path = PROJECT_ROOT / "data" / "results" / "lstm_direction_backtest.png"
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="LSTM plot not found")
    return FileResponse(chart_path)

# ---------- dashboard ----------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    s = PIPELINE_STATUS
    status_text = s["current_step"] or "idle"
    status_color = (
        "#FFA500" if s["is_running"]
        else "#4CAF50" if s["last_run_ok"]
        else "#F44336" if s["last_run_ok"] is False
        else "#9E9E9E"
    )
    last_start = s["last_run_start"] or "never"
    last_end = s["last_run_end"] or "-"
    last_error = s["last_error"] or "-"

    logs_html = "<br>".join(s["log"][-50:]) if s["log"] else "No logs yet."

    # Ensemble chart for dashboard
    chart_html = ""
    chart_path = PROJECT_ROOT / "data" / "results" / "ensemble_backtest.png"
    if chart_path.exists():
        with open(chart_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        chart_html = (
            f'<img src="data:image/png;base64,{img_data}" '
            f'style="width:100%; max-width:1200px; margin-top:20px;">'
        )

    # Results table
    results_table = ""
    results_path = PROJECT_ROOT / "data" / "results" / "strategy_comparison.csv"
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path)
        results_table = f"""
        <h3>Strategy Performance Comparison</h3>
        <div style="overflow-x:auto;">
            {df.to_html(index=False, classes='results-table', border=0)}
        </div>
        """

    return f"""
    <html>
    <head>
        <title>newalgotrade dashboard</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
            .card {{ background: white; border-radius: 8px; padding: 20px; max-width: 1400px; margin: 0 auto; }}
            .status-box {{ background: {status_color}; color: white; padding: 15px; border-radius: 6px; margin-bottom: 20px; }}
            .btn {{ display: inline-block; padding: 10px 18px; margin: 8px 4px 0 0;
                    background: #1976D2; color: white; text-decoration: none; border-radius: 4px; border: none; cursor: pointer; }}
            .btn:hover {{ background: #1565C0; }}
            .logs {{ margin-top: 20px; background: #212121; color: #e0e0e0;
                     padding: 10px; border-radius: 4px; font-family: monospace;
                     max-height: 300px; overflow-y: auto; font-size: 12px; }}
            .results-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            .results-table th {{ background: #1976D2; color: white; padding: 10px; text-align: left; }}
            .results-table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            .results-table tr:hover {{ background: #f5f5f5; }}
            .chart-container {{ margin-top: 30px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>newalgotrade pipeline</h1>
            <div class="status-box">
                <h2>Status: {status_text}</h2>
                <p><strong>Running:</strong> {s["is_running"]}</p>
                <p><strong>Last start:</strong> {last_start}</p>
                <p><strong>Last end:</strong> {last_end}</p>
                <p><strong>Last ok:</strong> {s["last_run_ok"]}</p>
                <p><strong>Last error:</strong> {last_error}</p>
            </div>

            <form action="/run-pipeline" method="post" style="display:inline;">
                <button type="submit" class="btn">Run pipeline now</button>
            </form>
            <a href="/logs" class="btn">Get logs (JSON)</a>
            <a href="/backtest-plot" class="btn">Ensemble plot</a>
            <a href="/lstm-plot" class="btn">LSTM plot</a>

            {results_table}

            <div class="chart-container">
                <h3>Backtest Results (Ensemble)</h3>
                {chart_html if chart_html else "<p style='color:#999;'>Chart will appear after pipeline completes</p>"}
            </div>

            <h3>Recent logs</h3>
            <div class="logs">{logs_html}</div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)