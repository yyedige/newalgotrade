# Algorithmic Trading

`Algorithmic Trading` is an experimental algorithmic trading project that uses an various models to forecast future returns and generate trading signals. This project was done under supervision of Issagali Konysbayev in a course "AI and ML in Finance". Use the project for educational uses.

---

## Getting started

To start the project, you should start the FastAPI in a file main.py. To see the results get URL from FastAPI 

### Prerequisites

- Python 3.10 or newer  
- Git
- A virtual environment tool (`venv`, `conda`, etc.)

---

## Configuration

All important settings live in `config.py`. A typical pattern is:

---

## Structure and Pipeline

The project is used from muny steps:
1. Data is loaded from yfinance in data/raw folder
2. The file is processed then converted to sqlite for better use and saved in folder data/processed
3. Models create a signals and signals are saved in folder data/signals
4. The signals generate a single signal nad that signal then is backtested


## Signal logic

`total_signal.py` contains the logic that turns model predictions into discrete trading decisions and computes the equity curve.

A typical pattern looks like this (you can adapt it to your real code):

---

## Project dependencies

The used libraries are in requirement.txt

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