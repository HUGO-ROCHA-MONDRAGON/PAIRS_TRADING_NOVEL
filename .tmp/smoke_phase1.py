from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel")
RAW = ROOT / "src" / "data" / "raw"

prices = pd.read_parquet(RAW / "prices.parquet").replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)
log_prices = np.log(prices)
pe = pd.read_parquet(RAW / "pe_ratios.parquet")
univ = pd.read_parquet(RAW / "universe.parquet")

prices.index = pd.to_datetime(prices.index)
pe.index = pd.to_datetime(pe.index)
univ["date"] = pd.to_datetime(univ["date"])

BACKTEST_START = pd.Timestamp("2017-01-01")
BACKTEST_END = pd.Timestamp("2018-12-31")
LOOKBACK_DAYS = 504
MIN_OBS = 252
MIN_COVERAGE = 0.90
MAX_TICKERS = 80
MAX_PAIRS = 15
ADF_T_THRESHOLD = -2.5

rebalance_dates = sorted(d for d in univ['date'].unique() if BACKTEST_START <= d <= BACKTEST_END)

def fit_ols(y, x):
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 1e-12:
        return np.nan, np.nan, np.array([])
    beta = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    alpha = y_mean - beta * x_mean
    resid = y - (alpha + beta * x)
    return alpha, beta, resid

def adf0_t_stat(series):
    e = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(e) < 30:
        return np.nan
    de = np.diff(e)
    lag = e[:-1]
    X = np.column_stack([np.ones_like(lag), lag])
    y = de
    try:
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta_hat
        dof = len(y) - X.shape[1]
        if dof <= 0:
            return np.nan
        s2 = float((resid @ resid) / dof)
        xtx_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt((s2 * xtx_inv)[1, 1]))
        if se <= 1e-12:
            return np.nan
        return float(beta_hat[1] / se)
    except np.linalg.LinAlgError:
        return np.nan

def score_pair(a, b):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < MIN_OBS:
        return None
    y1 = df.iloc[:,0].to_numpy(dtype=float)
    x1 = df.iloc[:,1].to_numpy(dtype=float)
    alpha1, beta1, r1 = fit_ols(y1, x1)
    t1 = adf0_t_stat(r1)
    y2 = df.iloc[:,1].to_numpy(dtype=float)
    x2 = df.iloc[:,0].to_numpy(dtype=float)
    alpha2, beta2, r2 = fit_ols(y2, x2)
    t2 = adf0_t_stat(r2)
    cand = []
    if not np.isnan(t1):
        cand.append(("a_on_b", alpha1, beta1, r1, t1))
    if not np.isnan(t2):
        cand.append(("b_on_a", alpha2, beta2, r2, t2))
    if not cand:
        return None
    orientation, alpha, beta, resid, t = min(cand, key=lambda x: x[4])
    sigma = float(np.std(resid, ddof=1))
    if not np.isfinite(sigma) or sigma <= 1e-8:
        return None
    if t >= ADF_T_THRESHOLD:
        return None
    return {"orientation": orientation, "alpha": alpha, "beta": beta, "mu": float(np.mean(resid)), "sigma": sigma, "t": float(t), "w": float(-t)}

processed = 0
months_with_pairs = 0
for reb, nxt in zip(rebalance_dates[:-1], rebalance_dates[1:]):
    formation = prices.index[prices.index <= reb]
    trade_end = prices.index[prices.index <= nxt]
    if len(formation)==0 or len(trade_end)==0:
        continue
    fe = formation.max()
    hist = log_prices.loc[:fe].tail(LOOKBACK_DAYS)
    active = set(univ.loc[univ['date']==reb, 'ticker'])
    cols = [c for c in hist.columns if c in active]
    if len(cols) < 20:
        continue
    coverage = hist[cols].notna().mean()
    eligible = coverage[coverage >= MIN_COVERAGE].index.tolist()
    if not eligible:
        continue
    pe_snap = pe.reindex(hist.index)[eligible].loc[:fe].ffill().iloc[-1]
    pe_snap = pe_snap[np.isfinite(pe_snap)]
    pe_snap = pe_snap[pe_snap > 0]
    candidates = pe_snap.sort_values().index.tolist()[:MAX_TICKERS]
    if len(candidates) < 10:
        continue
    records = []
    h = hist[candidates]
    tick = h.columns.tolist()
    for i, j in itertools.combinations(range(len(tick)), 2):
        s = score_pair(h[tick[i]], h[tick[j]])
        if s is None:
            continue
        records.append((tick[i], tick[j], s["w"]))
    processed += 1
    if records:
        months_with_pairs += 1

print(f"processed_rebalances={processed}")
print(f"months_with_candidate_pairs={months_with_pairs}")
