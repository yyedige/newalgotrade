# fastapi_app.py
from pathlib import Path
from datetime import datetime
import asyncio

from fastapi import FastAPI, BackgroundTasks
import papermill as pm

app = FastAPI(title="NEWALGOTRADE LSTM Pipeline")

PIPELINE_STATUS = {
    "last_run_start": None,
    "last_run_end": None,
    "last_run_ok": None,
    "last_error": None,
    "current_step": None,
    "log": [],          # list of text messages with timestamps
    "is_running": False,
}

# NEWALGOTRADE root (this file is here)
project_root = Path(__file__).resolve().parent

# Exact notebook locations
NOTEBOOK_PATHS = [
    project_root / "data_load.ipynb",            # in root
    project_root / "data_processer.ipynb",       # in root
    project_root / "model" / "lstm.ipynb",       # in model/
    project_root / "backtest.ipynb",             # in root
]

# Executed notebooks will be saved in NEWALGOTRADE/executed
executed_dir = project_root / "executed"
executed_dir.mkdir(exist_ok=True)
print("Executed notebooks will be saved to:", executed_dir)


def _log(msg: str):
    """Append a timestamped message to the in-memory log."""
    now = datetime.utcnow().isoformat()
    line = f"[{now}] {msg}"
    print(line)
    PIPELINE_STATUS["log"].append(line)


def run_pipeline():
    """Execute data_load -> data_processer -> lstm -> backtest."""
    global PIPELINE_STATUS

    if PIPELINE_STATUS.get("is_running"):
        _log("Previous run still in progress, skipping new run.")
        return

    PIPELINE_STATUS["is_running"] = True

    start_time = datetime.utcnow()
    PIPELINE_STATUS.update(
        {
            "last_run_start": start_time.isoformat(),
            "last_run_end": None,
            "last_run_ok": None,
            "last_error": None,
            "current_step": "starting",
            "log": [],
        }
    )
    _log("Pipeline run started")

    try:
        for nb_path in NOTEBOOK_PATHS:
            if not nb_path.exists():
                raise FileNotFoundError(f"Notebook not found: {nb_path}")

            PIPELINE_STATUS["current_step"] = f"running {nb_path.name}"
            _log(f"Executing notebook: {nb_path}")

            # Executed copy in NEWALGOTRADE/executed
            output_nb = executed_dir / f"{nb_path.stem}_executed.ipynb"

            # Pass PROJECT_ROOT so notebooks can build correct paths
            pm.execute_notebook(
                input_path=str(nb_path),
                output_path=str(output_nb),
                parameters={"PROJECT_ROOT": str(project_root)},
                progress_bar=False,
                report_mode=False,
            )

            _log(f"Finished notebook: {nb_path}")

        PIPELINE_STATUS["current_step"] = "idle"
        PIPELINE_STATUS["last_run_ok"] = True
        _log("Pipeline run finished successfully")

    except Exception as e:
        PIPELINE_STATUS["current_step"] = "error"
        PIPELINE_STATUS["last_run_ok"] = False
        PIPELINE_STATUS["last_error"] = repr(e)
        _log(f"Pipeline error: {repr(e)}")

    finally:
        PIPELINE_STATUS["last_run_end"] = datetime.utcnow().isoformat()
        PIPELINE_STATUS["is_running"] = False


# -------- periodic runner (every 60s) --------
async def periodic_runner():
    while True:
        try:
            run_pipeline()
        except Exception as e:
            _log(f"Unexpected error in periodic_runner: {repr(e)}")
        await asyncio.sleep(60)  # run every 60 seconds


@app.on_event("startup")
async def on_startup():
    """Start the periodic pipeline runner when FastAPI starts."""
    _log("Starting periodic runner (every 60 seconds)")
    asyncio.create_task(periodic_runner())


# Optional manual trigger endpoint
@app.post("/run-pipeline")
async def run_pipeline_endpoint(background_tasks: BackgroundTasks):
    """Manually start the full NVDA LSTM pipeline in the background."""
    background_tasks.add_task(run_pipeline)
    return {"message": "LSTM pipeline started"}


@app.get("/status")
async def status():
    """Check last pipeline run status and log."""
    return PIPELINE_STATUS


# -------- allow `python fastapi_app.py` --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )