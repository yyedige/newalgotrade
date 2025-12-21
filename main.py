from pathlib import Path
from datetime import datetime
import asyncio
import sys
import logging

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
    ("Data processing",  PROJECT_ROOT / "data" "data_processer.py"),
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
        for name, script in PIPELINE_STEPS:
            if not script.exists():
                raise FileNotFoundError(f"Script not found: {script}")

            PIPELINE_STATUS["current_step"] = name
            _log(f"Running step: {name} ({script.name})")

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if stdout:
                out_lines = stdout.decode(errors="ignore").strip().splitlines()
                if out_lines:
                    _log(out_lines[-1])

            if proc.returncode != 0:
                err = stderr.decode(errors="ignore")
                raise RuntimeError(f"{name} failed: {err}")

            _log(f"Finished step: {name}")

        PIPELINE_STATUS["last_run_ok"] = True
        _log("=== PIPELINE FINISHED OK ===")

    except Exception as e:
        PIPELINE_STATUS["last_run_ok"] = False
        PIPELINE_STATUS["last_error"] = str(e)
        _log(f"PIPELINE ERROR: {e}")

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
            await asyncio.sleep(180)  # 3 minutes

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

    return f"""
    <html>
    <head>
        <title>newalgotrade dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
            .card {{ background: white; border-radius: 8px; padding: 20px; max-width: 900px; margin: 0 auto; }}
            .status-box {{ background: {status_color}; color: white; padding: 15px; border-radius: 6px; }}
            .btn {{ display: inline-block; padding: 10px 18px; margin: 8px 4px 0 0;
                    background: #1976D2; color: white; text-decoration: none; border-radius: 4px; }}
            .btn:disabled {{ background: #9E9E9E; }}
            .logs {{ margin-top: 20px; background: #212121; color: #e0e0e0;
                     padding: 10px; border-radius: 4px; font-family: monospace;
                     max-height: 400px; overflow-y: auto; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>newalgotrade pipeline</h1>
            <div class="status-box">
                <h2>Status: {status_text}</h2>
                <p>Running: {s["is_running"]}</p>
                <p>Last start: {last_start}</p>
                <p>Last end: {last_end}</p>
                <p>Last ok: {s["last_run_ok"]}</p>
                <p>Last error: {last_error}</p>
            </div>
            <a href="/run-pipeline" class="btn">Run pipeline</a>
            <a href="/logs" class="btn">Get logs (JSON)</a>
            <h3>Recent logs</h3>
            <div class="logs">{logs_html}</div>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)