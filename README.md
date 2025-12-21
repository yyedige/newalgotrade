## Algorithmic Trading
Algorithmic Trading is an experimental trading framework that uses several machine learning models to forecast future returns and generate trading signals. This project was completed under the supervision of Issagali Konysbayev for the course “AI and ML in Finance” and is intended strictly for educational use.

## Getting started
To run the project, launch the FastAPI application defined in main.py. Once the server is running, you can view results and interact with the models by sending HTTP requests to the URLs exposed by the FastAPI endpoints (for example using a browser, curl, or tools like Postman).

## Prerequisites
Python 3.10 or newer
Git
A virtual environment tool (venv, conda, etc.)

## Basic workflow:

bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload      # or: python -m uvicorn main:app --reload

## Configuration
All key settings are centralized in config.py. Typical items configured there include:

Data paths (raw, processed, signals, results)

Universe of tickers and data frequency

Backtest window and transaction cost assumptions

Update config.py to point to your preferred tickers, date ranges, and model parameters before running the pipeline.

## Structure and pipeline
The project implements a multi‑step research and trading pipeline:

# Data acquisition

Raw market data is downloaded from yfinance and stored under data/raw.

# Data processing and storage

Raw files are cleaned, transformed and then written into an SQLite database for efficient querying.

Processed data is stored under data/processed.

# Signal generation by models

Multiple models are trained and/or applied on the processed data to generate individual trading signals.

These per‑model signals are saved under data/signals.

# Signal aggregation and backtesting

The individual signals are combined into a single aggregated signal that represents the final trading decision.

This combined signal is then backtested to produce performance statistics and equity curves.

# Signal logic
The file total_signal.py contains the logic that converts model outputs into executable trading decisions and computes the resulting equity curve. Typical responsibilities include:

Mapping model predictions into discrete positions (e.g. long / short / flat).

Aggregating signals from different models into one “total” signal.

## Project dependencies
All required Python libraries are listed in requirements.txt. To install them in your environment:

bash
pip install -r requirements.txt
This includes packages for data access (yfinance), database handling (SQLite interface), FastAPI for the web API, and the machine learning / backtesting stack.

## Roadmap
Planned and potential improvements include:

Clearer separation between training, inference, and backtesting flows.

Portfolio‑level backtesting across multiple symbols and asset classes.

Hyperparameter search and experiment tracking for model selection.

Additional model families (e.g. Transformers, CNNs, ensembles).

Richer performance and risk analytics, including factor and regime analysis.

## Disclaimer
This project is for research and educational purposes only and does not constitute investment advice.
No representation is made that any strategy implemented with this code will be profitable, and any use in live trading is solely at your own risk.