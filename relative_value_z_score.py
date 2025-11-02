"""
Relative-Value Dashboard (G10 FX, G3 10Y, Major Indices)
--------------------------------------------------------
• Discovers relationships with Graphical Lasso on returns (252d lookback by default)
• Validates with Engle–Granger cointegration on levels (logs for fx/indices, level p.p. for yields)
• Publishes per-pair residual Z-scores and per-anchor aggregate Z
• Built for Streamlit; swap the data adapter in `load_prices` to use your in-house source

Author: Wai Hin & Daron
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx

from dateutil.relativedelta import relativedelta
from sklearn.covariance import GraphicalLassoCV
from statsmodels.tsa.stattools import adfuller
import yfinance as yf


# -----------------------------
# ---------- CONFIG -----------
# -----------------------------
st.set_page_config(page_title="RV Dashboard: G10 FX / G3 Yields / Majors", layout="wide")

DEFAULT_LOOKBACK_DAYS = 800       # raw fetch
GLASSO_LOOKBACK = 252             # returns window for Graphical Lasso
COINT_LOOKBACK = 252              # levels window for cointegration
Z_WINDOW_DEFAULT = 126            # rolling window for Z-score
EWMA_HALFLIFE = 30                # if EWMA is chosen
ADF_CUTOFF_DEFAULT = 0.05

# -----------------------------
# ---------- UNIVERSE ---------
# -----------------------------
# Yahoo tickers for convenience. Replace with your vendor symbols if needed.
UNIVERSE = {
    "FX_G10": {
        # USD legs where possible; for NOK/SEK/Yen we include common crosses.
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "AUDUSD": "AUDUSD=X",
        "NZDUSD": "NZDUSD=X",
        "USDJPY": "JPY=X",        # note: JPY=X is USDJPY on Yahoo (price ~ 150 means 150 JPY per USD)
        "USDCHF": "CHF=X",
        "USDCAD": "CAD=X",
        "USDNOK": "NOK=X",
        "USDSEK": "SEK=X",
        # Optional: "USD" vs "DKK" "SGD" etc., but we keep strict G10.
    },
    "G3_Yields_10Y": {
        # Yields: use p.p. (percent points). ^TNX is US 10y *10; we’ll convert.
        "UST10Y": "^TNX",         # CBOE 10y yield (percentage*10)
        "Bund10Y": "DE10Y-BUND",  # <-- If unavailable on Yahoo, replace w/ your data source
        "JGB10Y": "JP10Y-JGB",    # <-- Same note as above
    },
    "Major_Indices": {
        "SPX": "^GSPC",
        "SX5E": "^STOXX50E",
        "NKY": "^N225",
        "FTSE100": "^FTSE",
        "HSI": "^HSI",
    }
}

# Fallback mapping for tickers that may not exist on Yahoo. You can point these to proxies or ETFs.
PROXY_TICKERS = {
    "Bund10Y": "IEGA.AS",  # (placeholder ETF proxy; swap to your rates source)
    "JGB10Y": "1312.T",    # (placeholder JGB ETF; swap to your rates source)
}

# For yield normalization
YIELD_TICKERS = {"^TNX"}  # add your real 10y yield tickers here if using Yahoo


# -----------------------------
# ---------- HELPERS ----------
# -----------------------------
@st.cache_data(show_spinner=False)
def load_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load Adj Close (or Close) prices from Yahoo Finance.
    Replace/extend this function to use your in-house data vendor.
    """
    # Resolve proxies
    y_tickers = []
    alias = {}
    for t in tickers:
        ysym = t
        # Convert our "semantic" keys to Yahoo tickers if this function is fed keys
        # Here, assume we're passing Yahoo tickers already; if not, look up from UNIVERSE.
        y_tickers.append(ysym)
        alias[ysym] = ysym

    df = yf.download(y_tickers, start=start, end=end, progress=False)
    # Try Adj Close then Close
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.levels[0]):
            px = df["Adj Close"].copy()
        elif ("Close" in df.columns.levels[0]):
            px = df["Close"].copy()
        else:
            px = df.xs(df.columns.levels[0][0], axis=1, level=0)  # first field fallback
    else:
        px = df.copy()

    px = px.ffill().dropna(how="all")
    return px


def normalize_levels(raw_px: pd.DataFrame, col_meta: dict[str, str]) -> pd.DataFrame:
    """
    Normalize to level series used for cointegration.
    - FX & Indices: log price
    - Yields: percentage points (not logs)
    col_meta maps column -> 'yield' | 'price'
    """
    out = pd.DataFrame(index=raw_px.index)
    for c in raw_px.columns:
        if col_meta.get(c, "price") == "yield":
            # ^TNX is in “percentage*10”. Convert to percent points.
            if c in YIELD_TICKERS:
                out[c] = raw_px[c] / 10.0  # eg 45.3 -> 4.53 p.p.
            else:
                # If your source is already in p.p., pass through:
                out[c] = raw_px[c].astype(float)
        else:
            out[c] = np.log(raw_px[c].astype(float))
    return out


def simple_returns(level_df: pd.DataFrame, col_meta: dict[str, str]) -> pd.DataFrame:
    """
    Return series used for covariance/Graphical Lasso.
    - For price-like series: log returns (diff of logs)
    - For yields: simple changes in p.p.
    """
    rets = pd.DataFrame(index=level_df.index)
    for c in level_df.columns:
        if col_meta.get(c, "price") == "yield":
            rets[c] = level_df[c].diff()
        else:
            rets[c] = level_df[c].diff()
    return rets.dropna()


def build_col_meta(columns: list[str]) -> dict[str, str]:
    """
    Tag each column as 'yield' or 'price' for normalization logic.
    """
    meta = {}
    for c in columns:
        if c in YIELD_TICKERS:
            meta[c] = "yield"
        else:
            meta[c] = "price"
    return meta


def adf_pvalue(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 30:
        return np.nan
    try:
        return adfuller(s, regression="c", autolag="AIC")[1]
    except Exception:
        return np.nan


def engle_granger_levels(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, pd.Series, float]:
    """
    EG 2-step on levels: x = a + b*y + e; ADF on e
    Returns (beta=[a,b], resid, adf_p)
    """
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 40:
        return np.array([np.nan, np.nan]), pd.Series(index=df.index, dtype=float), np.nan
    X = np.c_[np.ones(len(df)), df.iloc[:, 1].values]
    y_vec = df.iloc[:, 0].values
    beta, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
    resid = df.iloc[:, 0] - (beta[0] + beta[1] * df.iloc[:, 1])
    pval = adf_pvalue(resid)
    resid.name = f"resid({df.columns[0]}~{df.columns[1]})"
    return beta, resid, pval


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd


def ewma_z(series: pd.Series, halflife: int) -> pd.Series:
    mu = series.ewm(halflife=halflife, adjust=False).mean()
    sd = series.ewm(halflife=halflife, adjust=False).std()
    return (series - mu) / sd


def neighbors_of(target: str, prec: pd.DataFrame, thr: float = 1e-8) -> list[str]:
    row = prec.loc[target].drop(target)
    return list(row[np.abs(row) > thr].index)


def to_graph_from_precision(prec: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    cols = list(prec.columns)
    G.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            w = prec.loc[a, b]
            if abs(w) > 1e-8:
                G.add_edge(a, b, weight=float(w))
    return G


# -----------------------------
# ---------- SIDEBAR ----------
# -----------------------------
st.sidebar.header("Controls")

# Date range
end_date = pd.to_datetime("today").normalize()
start_date = end_date - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS)
start = st.sidebar.date_input("Start", start_date)
end = st.sidebar.date_input("End", end_date)

# Windows & thresholds
z_window = st.sidebar.number_input("Z-score window", 30, 300, Z_WINDOW_DEFAULT)
use_ewma = st.sidebar.checkbox("Use EWMA for Z (halflife=30)", value=False)
adf_cut = st.sidebar.slider("ADF p-value cutoff", 0.01, 0.20, ADF_CUTOFF_DEFAULT)
glasso_lookback = st.sidebar.number_input("Graphical Lasso lookback (days)", 100, 750, GLASSO_LOOKBACK)
coint_lookback = st.sidebar.number_input("Cointegration lookback (days)", 100, 750, COINT_LOOKBACK)

# Universe pickers
st.sidebar.subheader("Universe")
use_fx = st.sidebar.checkbox("Include G10 FX", True)
use_rates = st.sidebar.checkbox("Include G3 10Y Yields", True)
use_eq = st.sidebar.checkbox("Include Major Indices", True)

# Anchor selection (assets for which to publish an Aggregate Z)
st.sidebar.subheader("Anchors (publish Aggregate Z)")
default_fx_anchors = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X"]
default_eq_anchors = ["^GSPC", "^STOXX50E"]
default_rate_anchors = ["^TNX"]

# Build the full list of Yahoo tickers & label map
def collect_tickers() -> tuple[list[str], dict[str, str]]:
    tickers = []
    label_map = {}  # human label -> yahoo ticker
    if use_fx:
        for k, v in UNIVERSE["FX_G10"].items():
            label_map[k] = v
            tickers.append(v)
    if use_rates:
        for k, v in UNIVERSE["G3_Yields_10Y"].items():
            v2 = PROXY_TICKERS.get(k, v)
            label_map[k] = v2
            tickers.append(v2)
    if use_eq:
        for k, v in UNIVERSE["Major_Indices"].items():
            label_map[k] = v
            tickers.append(v)
    # De-dup
    tickers = sorted(set(tickers))
    return tickers, label_map

all_tickers, label_map = collect_tickers()

anchors = st.sidebar.multiselect(
    "Anchors",
    options=all_tickers,
    default=[t for t in all_tickers if t in (default_fx_anchors + default_eq_anchors + default_rate_anchors)]
)

# -----------------------------
# ---------- DATA -------------
# -----------------------------
if len(all_tickers) == 0:
    st.error("No universe selected.")
    st.stop()

px_raw = load_prices(all_tickers, pd.to_datetime(start), pd.to_datetime(end))
if px_raw.empty or px_raw.shape[1] == 0:
    st.error("No data returned. Check tickers/data source.")
    st.stop()

# Column metadata: tag yields vs prices
col_meta = build_col_meta(list(px_raw.columns))

# Normalize to cointegration levels (logs for prices; p.p. for yields)
levels = normalize_levels(px_raw, col_meta)

# Return series for Graphical Lasso
rets = simple_returns(levels, col_meta)
rets_std = (rets - rets.mean()) / rets.std()
rets_std = rets_std.dropna()

# Truncate lookbacks
rets_lkb = rets_std.iloc[-glasso_lookback:].dropna(how="any")
levels_lkb = levels.iloc[-coint_lookback:].dropna(how="any")

st.title("Relative-Value Discovery & Z-Score Dashboard")
st.caption("G10 FX, G3 10Y, Major Indices • Graphical Lasso → Cointegration → Residual Z-scores")

# -----------------------------
# ----- GRAPHICAL LASSO -------
# -----------------------------
with st.expander("Graphical Lasso (Discovery) — click to view details", expanded=True):
    st.write(
        f"Fitting GraphicalLassoCV on standardized returns (lookback: **{glasso_lookback}** days, N={rets_lkb.shape[1]})."
    )
    try:
        gl = GraphicalLassoCV().fit(rets_lkb.values)
        prec = pd.DataFrame(gl.precision_, index=rets_lkb.columns, columns=rets_lkb.columns)
        st.success("Graphical Lasso fit successful.")
    except Exception as e:
        st.error(f"Graphical Lasso failed: {e}")
        st.stop()

    # Optional: show degree for each node
    deg = (np.abs(prec) > 1e-8).sum(axis=1) - 1
    deg_df = deg.sort_values(ascending=False).to_frame("degree")
    st.dataframe(deg_df, use_container_width=True, height=220)

    # Optional: network view (edge list)
    edges = []
    cols = list(prec.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            w = prec.loc[a, b]
            if abs(w) > 1e-8:
                edges.append({"u": a, "v": b, "weight": float(w)})
    st.write(f"Edges discovered: {len(edges)}")
    if len(edges) > 0:
        st.dataframe(pd.DataFrame(edges).sort_values("weight", key=np.abs, ascending=False), use_container_width=True, height=240)

# -----------------------------
# ----- COINTEGRATION + Z -----
# -----------------------------
def compute_pair_metrics(lhs: str, rhs: str) -> dict:
    """
    For a candidate pair (lhs vs rhs):
    - Engle–Granger on levels
    - Residual z-score (rolling or EWMA)
    - Return dict for table
    """
    df = levels_lkb[[lhs, rhs]].dropna()
    if len(df) < 40:
        return {}

    beta, resid, pval = engle_granger_levels(df[lhs], df[rhs])
    if np.isnan(pval):
        return {}

    if use_ewma:
        z = ewma_z(resid, EWMA_HALFLIFE)
    else:
        z = rolling_z(resid, z_window)

    last_z = float(z.dropna().iloc[-1]) if z.dropna().shape[0] else np.nan
    last_spread = float(resid.dropna().iloc[-1]) if resid.dropna().shape[0] else np.nan

    return {
        "lhs": lhs,
        "rhs": rhs,
        "beta": float(beta[1]),
        "alpha": float(beta[0]),
        "adf_p": float(pval),
        "z": last_z,
        "spread_last": last_spread,
    }


def discover_neighbors_for(anchor: str) -> list[str]:
    if anchor not in prec.index:
        return []
    return neighbors_of(anchor, prec, thr=1e-8)


# Build pair list: for each selected anchor, take its Graphical Lasso neighbors
pair_rows = []
for a in anchors:
    neigh = discover_neighbors_for(a)
    for b in neigh:
        # Create ordered pair (a vs b)
        pair_rows.append((a, b))

pair_rows = sorted(set(pair_rows))
st.subheader("Candidates from Graphical Lasso (by anchor)")
st.caption("Only pairs touching selected anchors are tested for cointegration & z-score.")
st.write(", ".join([f"({a} vs {b})" for a, b in pair_rows]) or "—")

# Compute metrics per pair; keep those passing cointegration
records = []
for (lhs, rhs) in pair_rows:
    rec = compute_pair_metrics(lhs, rhs)
    if not rec:
        continue
    records.append(rec)

pairs_df = pd.DataFrame(records)
if pairs_df.empty:
    st.warning("No valid pairs computed (insufficient data or model failure).")
else:
    pairs_df["passes_coint"] = pairs_df["adf_p"] <= adf_cut
    st.subheader("Pair Results")
    st.dataframe(
        pairs_df.sort_values(["passes_coint", "z"], ascending=[False, True]),
        use_container_width=True,
        height=300
    )

# -----------------------------
# ----- AGGREGATE Z (ANCHOR) ---
# -----------------------------
st.subheader("Anchor Aggregate Z-Scores")
st.caption("Average of pair Z for pairs that **pass** cointegration and touch the anchor as LHS.")

anchor_rows = []
if not pairs_df.empty:
    for a in anchors:
        sub = pairs_df[(pairs_df["lhs"] == a) & (pairs_df["passes_coint"])]
        agg = float(sub["z"].mean()) if len(sub) else np.nan
        anchor_rows.append({"anchor": a, "aggregate_z": agg, "num_pairs": int(len(sub))})

agg_df = pd.DataFrame(anchor_rows)
if agg_df.empty:
    st.write("—")
else:
    st.dataframe(agg_df.sort_values("aggregate_z"), use_container_width=True, height=200)

# KPI cards
cols = st.columns(min(4, max(1, len(anchor_rows))))
for i, row in enumerate(anchor_rows[:len(cols)]):
    with cols[i]:
        val = "—" if np.isnan(row["aggregate_z"]) else f"{row['aggregate_z']:.2f}"
        st.metric(f"{row['anchor']} Aggregate Z", val, help=f"Based on {row['num_pairs']} passing pairs")

# -----------------------------
# ----- INSPECT A PAIR --------
# -----------------------------
if not pairs_df.empty:
    st.subheader("Inspect a Pair (spread & ±2σ)")
    passing = pairs_df[pairs_df["passes_coint"]].copy()
    if passing.empty:
        st.info("No passing pairs. Relax ADF cutoff or extend lookback.")
    else:
        passing["pair_name"] = passing["lhs"] + " vs " + passing["rhs"]
        pick = st.selectbox("Select pair", options=passing["pair_name"], index=0)
        lhs = pick.split(" vs ")[0]
        rhs = pick.split(" vs ")[1]

        # Rebuild residual and bands for the chart
        df = levels_lkb[[lhs, rhs]].dropna()
        beta, resid, pval = engle_granger_levels(df[lhs], df[rhs])
        if use_ewma:
            mu = resid.ewm(halflife=EWMA_HALFLIFE, adjust=False).mean()
            sd = resid.ewm(halflife=EWMA_HALFLIFE, adjust=False).std()
        else:
            mu = resid.rolling(z_window).mean()
            sd = resid.rolling(z_window).std()
        hi = mu + 2 * sd
        lo = mu - 2 * sd

        chart_df = pd.concat({"spread": resid, "+2σ": hi, "-2σ": lo}, axis=1).dropna()
        st.line_chart(chart_df)

        st.write(
            f"**β (hedge ratio)**: {beta[1]:.4f} | **ADF p**: {pval:.4f} | "
            f"**Latest Z**: {((resid - mu) / sd).dropna().iloc[-1]:.2f}"
        )

# -----------------------------
# ----- DIAGNOSTICS -----------
# -----------------------------
with st.expander("Diagnostics"):
    st.markdown(
        f"""
        **Lookbacks:** Graphical Lasso={glasso_lookback}d • Cointegration={coint_lookback}d • Z={z_window}d ({'EWMA' if use_ewma else 'Rolling'})  
        **ADF cutoff:** {adf_cut:.2f}  
        **Data rows:** returns={rets_lkb.shape[0]} • levels={levels_lkb.shape[0]}  
        """
    )
    st.write("Columns tagged as yields:", [c for c, t in build_col_meta(levels.columns).items() if t == "yield"])
    st.write("Columns tagged as prices:", [c for c, t in build_col_meta(levels.columns).items() if t == "price"])

    st.caption("Tip: If you see many failures, widen anchors, relax ADF to 0.10, or extend lookbacks.")


# -----------------------------
# ----- NOTES / TODO ----------
# -----------------------------
st.markdown(
    """
**Notes**
- **Interpretation**: Z > 0 ⇒ LHS asset is *rich* vs β-hedged RHS; Z < 0 ⇒ *cheap*.  
- **Aggregation**: Anchor Aggregate Z = mean of pair z-scores for (anchor vs neighbor) that pass cointegration.  
- **Scaling**: Yields are in percent points; indices/FX are log prices for cointegration, log-diff for returns.  
- **Stability**: Hedge ratio β is estimated on the cointegration window; consider expanding windows for stability.

**TODO (optional)**
- Replace Yahoo proxies for Bund/JGB with your rates feed (level p.p. series).
- Add alerts (|Z| ≥ 2 or 3).
- Add Johansen to form multi-asset EUR (or anchor) baskets.
- Persist last run and serve via Streamlit Cloud / internal infra.
"""
)
