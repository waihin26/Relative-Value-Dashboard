# Relative-Value Dashboard

Streamlit app to discover and visualise relative-value relationships across G10 FX, G3 10Y yields and major equity indices.

Key ideas
- Discover candidate relationships using Graphical Lasso on standardized returns (default 252-day lookback).
- Validate candidate pairs with Engle–Granger cointegration on levels (logs for FX/indices, percent-points for yields).
- Publish residual z-scores per pair and aggregate Z per chosen anchor.

Author: Wai Hin & Daron

Files
- `relative_value_z_score.py` — main Streamlit application. Replace `load_prices` inside it to connect to a different data vendor if needed.

Requirements
- Python 3.9+ recommended
- Packages (install from `requirements.txt`): streamlit, numpy, pandas, networkx, scikit-learn, statsmodels, yfinance, python-dateutil

Quick start
1. Create and activate a virtual environment (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run relative_value_z_score.py
```

3. In the sidebar you can choose the date range, lookbacks, universe toggles (FX / Yields / Indices) and anchors. The app will:
- Fit Graphical Lasso on returns to propose neighbors
- Run Engle–Granger on each candidate pair and compute residual Z-scores
- Show per-pair metrics and per-anchor aggregate Z

Notes and customization
- The included universe uses Yahoo Finance tickers and simple ETF proxies for some rates tickers. Replace `UNIVERSE`, `PROXY_TICKERS` or the implementation of `load_prices()` in `relative_value_z_score.py` to plug in your in-house data.
- Yields are treated in percentage points (p.p.). The code converts some Yahoo yield tickers (e.g., `^TNX`) to p.p.
- The default approach estimates pair hedge ratio β by OLS on levels (Engle–Granger). Consider using Johansen tests for multi-asset cointegration if you need basket construction.


