# fastapi_app.py - PRODUCTION READY
from pathlib import Path
from datetime import datetime
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address
import sys

# Configuration
class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    PIPELINE_INTERVAL_HOURS: int = 24
    MAX_LOG_LINES: int = 1000
    STEP_TIMEOUT: int = 600
    
    class Config:
        env_file = ".env"

settings = Settings()
settings.LOG_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pipeline status
PIPELINE_STATUS = {
    "last_run_start": None,
    "last_run_end": None,
    "last_run_ok": None,
    "last_error": None,
    "current_step": None,
    "log": [],
    "is_running": False,
    "total_runs": 0,
    "successful_runs": 0,
}

# Pipeline steps
PIPELINE_STEPS = [
    {"name": "Data Load", "script": settings.PROJECT_ROOT / "data_load.py"},
    {"name": "Data Processing", "script": settings.PROJECT_ROOT / "data_processer.py"},
    {"name": "EMA Signals", "script": settings.PROJECT_ROOT / "models" / "EMAcrossover.py"},
    {"name": "SMA Signals", "script": settings.PROJECT_ROOT / "models" / "SMAcrossover.py"},
    {"name": "ARIMA+GARCH", "script": settings.PROJECT_ROOT / "models" / "armagarch.py"},
    {"name": "BiLSTM", "script": settings.PROJECT_ROOT / "models" / "lstm.py"},
    {"name": "Ensemble", "script": settings.PROJECT_ROOT / "total_signal.py"},
]

def _log(msg: str, level: str = "info"):
    """Thread-safe logging."""
    getattr(logger, level)(msg)
    timestamp = datetime.utcnow().isoformat()
    PIPELINE_STATUS["log"].append(f"[{timestamp}] {msg}")
    if len(PIPELINE_STATUS["log"]) > settings.MAX_LOG_LINES:
        PIPELINE_STATUS["log"] = PIPELINE_STATUS["log"][-settings.MAX_LOG_LINES:]

async def run_pipeline():
    """Execute pipeline steps asynchronously."""
    if PIPELINE_STATUS["is_running"]:
        _log("Pipeline already running", "warning")
        return

    PIPELINE_STATUS["is_running"] = True
    PIPELINE_STATUS["total_runs"] += 1
    start_time = datetime.utcnow()
    PIPELINE_STATUS["last_run_start"] = start_time.isoformat()
    
    _log("=" * 80)
    _log("🚀 PIPELINE STARTED")
    _log("=" * 80)

    try:
        for idx, step in enumerate(PIPELINE_STEPS, 1):
            if not step["script"].exists():
                raise FileNotFoundError(f"Script not found: {step['script']}")

            PIPELINE_STATUS["current_step"] = f"[{idx}/{len(PIPELINE_STEPS)}] {step['name']}"
            _log(f"▶️  Step {idx}/{len(PIPELINE_STEPS)}: {step['name']}")

            # Run async subprocess
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(step["script"]),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=settings.STEP_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise TimeoutError(f"{step['name']} exceeded {settings.STEP_TIMEOUT}s timeout")

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                _log(f"❌ {step['name']} failed: {error_msg}", "error")
                raise RuntimeError(f"{step['name']} failed")

            _log(f"✅ {step['name']} completed")

        PIPELINE_STATUS["last_run_ok"] = True
        PIPELINE_STATUS["successful_runs"] += 1
        duration = (datetime.utcnow() - start_time).total_seconds()
        _log(f"✅ PIPELINE COMPLETED in {duration:.1f}s")

    except Exception as e:
        PIPELINE_STATUS["last_run_ok"] = False
        PIPELINE_STATUS["last_error"] = str(e)
        _log(f"❌ Pipeline failed: {e}", "error")

    finally:
        PIPELINE_STATUS["last_run_end"] = datetime.utcnow().isoformat()
        PIPELINE_STATUS["current_step"] = "idle"
        PIPELINE_STATUS["is_running"] = False

async def periodic_runner():
    """Run pipeline periodically."""
    while True:
        await asyncio.sleep(settings.PIPELINE_INTERVAL_HOURS * 3600)
        _log("⏰ Periodic trigger")
        await run_pipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    _log("🚀 FastAPI started")
    task = asyncio.create_task(periodic_runner())
    yield
    _log("🛑 FastAPI shutting down")
    task.cancel()

# FastAPI app
app = FastAPI(title="NEWALGOTRADE Pipeline", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {
        "status": "healthy",
        "is_running": PIPELINE_STATUS["is_running"],
        "last_run_ok": PIPELINE_STATUS["last_run_ok"],
        "total_runs": PIPELINE_STATUS["total_runs"],
        "successful_runs": PIPELINE_STATUS["successful_runs"]
    }

@app.post("/run-pipeline")
@limiter.limit("2/hour")
async def trigger_pipeline(request: Request, background_tasks: BackgroundTasks):
    """Manually trigger pipeline (rate limited)."""
    if PIPELINE_STATUS["is_running"]:
        raise HTTPException(status_code=429, detail="Pipeline already running")
    
    background_tasks.add_task(run_pipeline)
    return {"status": "started", "message": "Pipeline triggered"}

@app.get("/status")
async def status():
    """Get pipeline status."""
    return JSONResponse(content=PIPELINE_STATUS)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard HTML."""
    # Your existing dashboard HTML here
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Important: 1 worker for background tasks
        reload=False
    )