from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel")
RAW = ROOT / "src" / "data" / "raw"

prices = pd.read_parquet(RAW / "prices.parquet").replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()
log_prices = np.log(prices)

pe = pd.read_parquet(RAW / "pe_ratios.parquet")
pe.index = pd.to_datetime(pe.index)
pe = pe.sort_index()

univ = pd.read_parquet(RAW / "universe.parquet")
univ["date"] = pd.to_datetime(univ["date"])
univ = univ.sort_values(["date", "ticker"])

BACKTEST_START = pd.Timestamp("2017-01-01")
BACKTEST_END = pd.Timestamp("2017-12-31")
LOOKBACK_DAYS = 504
MIN_OBS = 252
MIN_COVERAGE = 0.90
MAX_TICKERS = 60
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
    vals = [v for v in [t1, t2] if not np.isnan(v)]
    if not vals:
        return None
    t = min(vals)
    if t >= ADF_T_THRESHOLD:
        return None
    return t

processed = 0
months_with_pairs = 0
for reb, nxt in zip(rebalance_dates[:-1], rebalance_dates[1:]):
    formation_candidates = prices.index[prices.index <= reb]
    if len(formation_candidates) == 0:
        continue
    formation_end = formation_candidates.max()

    hist = log_prices[log_prices.index <= formation_end].tail(LOOKBACK_DAYS)
    active = set(univ.loc[univ['date']==reb, 'ticker'])
    cols = [c for c in hist.columns if c in active]
    if len(cols) < 20:
        continue

    h = hist[cols]
    cov = h.notna().mean()
    eligible = cov[cov >= MIN_COVERAGE].index.tolist()
    if not eligible:
        continue

    pe_eligible = [t for t in eligible if t in pe.columns]
    if pe_eligible:
        pe_slice = pe.reindex(h.index)[pe_eligible]
        pe_hist = pe_slice[pe_slice.index <= formation_end]
        if pe_hist.empty:
            pe_snap = pd.Series(dtype=float)
        else:
            pe_snap = pe_hist.ffill().iloc[-1]
        pe_snap = pe_snap[np.isfinite(pe_snap)]
        pe_snap = pe_snap[pe_snap > 0]
        candidates = pe_snap.sort_values().index.tolist()
    else:
        candidates = []

    if len(candidates) < MAX_TICKERS:
        rem = [t for t in cov.sort_values(ascending=False).index if t in eligible and t not in candidates]
        candidates = (candidates + rem)[:MAX_TICKERS]

    if len(candidates) < 10:
        continue

    processed += 1
    found = 0
    tick = candidates
    for i, j in itertools.combinations(range(len(tick)), 2):
        t = score_pair(h[tick[i]], h[tick[j]])
        if t is not None:
            found += 1
    if found > 0:
        months_with_pairs += 1

print(f"processed_rebalances={processed}")
print(f"months_with_candidate_pairs={months_with_pairs}")
