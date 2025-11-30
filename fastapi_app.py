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

NOTEBOOK_ORDER = [
    "data_load.ipynb",
    "data_processer.ipynb",
    "lstm.ipynb",
    "backtest.ipynb",
]


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

    project_root = Path(__file__).resolve().parent      # NEWALGOTRADE
    model_dir = project_root / "model"
    executed_dir = model_dir / "executed"
    executed_dir.mkdir(exist_ok=True)

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
        for nb_name in NOTEBOOK_ORDER:
            input_nb = model_dir / nb_name
            output_nb = executed_dir / f"{input_nb.stem}_executed.ipynb"

            if not input_nb.exists():
                raise FileNotFoundError(f"Notebook not found: {input_nb}")

            PIPELINE_STATUS["current_step"] = f"running {nb_name}"
            _log(f"Executing notebook: {input_nb}")

            pm.execute_notebook(
                input_path=str(input_nb),
                output_path=str(output_nb),
                parameters={},
                progress_bar=False,
                report_mode=False,
            )

            _log(f"Finished notebook: {input_nb}")

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


# Optional manual trigger endpoint (kept if you still want it)
@app.post("/run-pipeline")
async def run_pipeline_endpoint(background_tasks: BackgroundTasks):
    """Manually start the full NVDA LSTM pipeline in the background."""
    background_tasks.add_task(run_pipeline)
    return {"message": "LSTM pipeline started"}


@app.get("/status")
async def status():
    """Check last pipeline run status and log."""
    return PIPELINE_STATUS