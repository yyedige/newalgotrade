# newalgotrade

`newalgotrade` is an experimental algorithmic trading project that uses an LSTM / BiLSTM model to forecast future returns and generate trading signals.  

---


You can refine the comments later as you add more folders (e.g. `data/raw`, `data/results`).

---

## Getting started

### Prerequisites

- Python 3.10 or newer  
- Git  
- A virtual environment tool (`venv`, `conda`, etc.)

### Installation


---

## Configuration

All important settings live in `config.py`. A typical pattern is:


Adjust names and values to match how you actually coded `main.py` and `total_signal.py`.

---

## Usage

### 1. Prepare data

Put your market data files under `data/`, for example:


Typical behaviour:

1. Load settings from `config.py`.  
2. Load and preprocess data from `DATA_DIR`.  
3. Build and train the BiLSTM using `MODEL_PARAMS`.  
4. Save the best model weights to `best_bilstm_future.pth` (and/or inside `models/`).

If your script uses a different flag (e.g. `--train`), just update this command.

### 3. Run a backtest

After `best_bilstm_future.pth` exists, run:


The backtest usually:

1. Loads configuration from `config.py`.  
2. Loads data for the backtest period.  
3. Loads the trained model weights from `best_bilstm_future.pth`.  
4. Produces predictions (future returns / prices).  
5. Calls functions in `total_signal.py` to map predictions to positions.  
6. Simulates trading with commission and slippage.  
7. Saves outputs under `data/results/` (e.g. `equity_curves.png`, CSVs of trades and returns).

---

## Signal logic

`total_signal.py` contains the logic that turns model predictions into discrete trading decisions and computes the equity curve.

A typical pattern looks like this (you can adapt it to your real code):


Replace this example with the actual logic you use, or keep it as a starting point.

---

## Project dependencies

Create a `requirements.txt` with something like:


---

## Roadmap

- Better separation of training and backtesting flows.  
- Portfolio‑level backtesting across multiple symbols.  
- Hyperparameter search and experiment tracking.  
- Additional models (Transformers, CNNs, ensembles).  
- More detailed performance reports and risk analytics.

---

## Disclaimer

This project is for research and educational purposes only and does **not** constitute investment advice.  
Use at your own risk. Past performance of simulated strategies does not guarantee future results.
