import yfinance as yf
import pandas as pd
import os
from pathlib import Path

# Get project root (parent of data folder)
BASE_DIR = Path(__file__).parent.parent
config_path = BASE_DIR / "config.py"

# Load config dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

TICKER = config.TICKER
DATA_START_DATE = config.DATA_START_DATE
DATA_END_DATE = config.DATA_END_DATE

# Download data
data = yf.download(tickers=TICKER, start=DATA_START_DATE, end=DATA_END_DATE)

# Save
out_dir = BASE_DIR / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)
data.to_csv(out_dir / "data.csv")