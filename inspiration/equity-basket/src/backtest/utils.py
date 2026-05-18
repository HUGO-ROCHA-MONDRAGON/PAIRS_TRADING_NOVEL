"""
Ensemble de fonctions utilisées dans les autres py: normalisation des tickers, requêtes univers point-in-time,
calendrier de rebalancement, calcul du beta, scoring du backtest.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


def normalize_ticker(ticker: str, price_columns: set) -> str:
    """Match le ticker à la convention de nommage des colonnes de prix."""
    t = str(ticker).strip()
    if t in price_columns:
        return t
    t_eq = f"{t} Equity"
    if t_eq in price_columns:
        return t_eq
    if t.endswith(" Equity") and t[:-7] in price_columns:
        return t[:-7]
    return t


def normalize_bbg_ticker(ticker: str, default_exchange: str = "US") -> str:
    """Ajoute le suffixe Bloomberg ' Equity' si absent."""
    t = str(ticker).strip()
    if t.endswith(" Equity"):
        return t
    if " " in t:
        return f"{t} Equity"
    return f"{t} {default_exchange} Equity"


def force_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Assure que 'date' est une colonne (pas l'index)."""
    if df is None:
        return pd.DataFrame(columns=["date", "ticker"])
    if "date" not in df.columns:
        if df.index.name == "date":
            df = df.reset_index()
        else:
            raise KeyError(f"Colonne 'date' manquante")
    df["date"] = pd.to_datetime(df["date"])
    return df


def constituents_to_long(constituents: pd.DataFrame) -> pd.DataFrame:
    """Convertit l'univers au format long [date, ticker]."""
    if constituents is None or len(constituents) == 0:
        return pd.DataFrame(columns=["date", "ticker"])

    if "date" in constituents.columns and "ticker" in constituents.columns:
        df = constituents.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.strip()
        return df[["date", "ticker"]]

    mat = constituents.copy()
    mat.index = pd.to_datetime(mat.index)
    long_df = (
        mat.fillna(0)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker", 0: "member"})
    )
    long_df = long_df[long_df["member"] > 0][["date", "ticker"]]
    long_df["ticker"] = long_df["ticker"].astype(str).str.strip()
    return long_df


def normalize_universe_tickers(
    universe_long: pd.DataFrame, price_columns
) -> pd.DataFrame:
    """Match les tickers univers avec les colonnes de prix."""
    if universe_long is None or universe_long.empty:
        return universe_long
    pcols = set(str(c).strip() for c in price_columns)
    out = universe_long.copy()
    out["ticker"] = out["ticker"].map(lambda t: normalize_ticker(t, pcols))
    return out


def universe_at_date(
    universe_long: pd.DataFrame, date: pd.Timestamp
) -> pd.Series:
    """Tickers de l'univers à une date donnée (point-in-time)."""
    if universe_long is None or universe_long.empty:
        return pd.Series(dtype=str)
    ul = universe_long.copy()
    ul["date"] = pd.to_datetime(ul["date"])
    eligible = ul["date"].unique()  # dates disponibles
    eligible = eligible[eligible <= pd.Timestamp(date)]
    if len(eligible) == 0:
        return pd.Series(dtype=str)
    use = pd.Timestamp(sorted(eligible)[-1])
    return ul.loc[ul["date"] == use, "ticker"].astype(str)


def universe_at_date_list(
    universe_long: pd.DataFrame, date: pd.Timestamp
) -> list:
    """Liste des tickers de l'univers à une date."""
    ul = universe_long.copy()
    ul["date"] = pd.to_datetime(ul["date"])
    e = ul.loc[ul["date"] <= date, "date"]
    if e.empty:
        return []
    return ul.loc[ul["date"] == e.max(), "ticker"].tolist()


def get_rebalance_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str = "ME",
) -> List[pd.Timestamp]:
    """Dates de rebalancement (default mensuel)."""
    return list(pd.date_range(start=start_date, end=end_date, freq=frequency))


def compute_rolling_beta(
    track: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Beta glissant de la stratégie vs benchmark."""
    sr = track.set_index("date")["net_return"]
    br = benchmark.pct_change().reindex(sr.index).fillna(0)
    # cov / var = beta
    return sr.rolling(window).cov(br) / br.rolling(window).var().replace(0, np.nan)


def compute_stock_betas(
    returns_window: pd.DataFrame,
    benchmark_returns: np.ndarray,
    tickers: list,
) -> np.ndarray:
    """Calcule les betas individuels des actions."""
    mv = np.var(benchmark_returns)
    if mv < 1e-12:
        return np.zeros(len(tickers))
    return np.array([
        np.cov(returns_window[tk].fillna(0).values, benchmark_returns)[0, 1] / mv
        if tk in returns_window.columns else 0.0
        for tk in tickers
    ])


def score_run(
    track: pd.DataFrame,
    benchmark: pd.Series,
    rf_mean: float,
    beta_window: int = 63,
) -> Dict[str, float]:
    """Calcule les métriques du backtest (Sharpe, Sortino, Calmar, etc.)."""
    rets = track["net_return"].values
    nd = len(rets)
    ny = nd / 252  # années
    tr = track["net_value"].iloc[-1] / track["net_value"].iloc[0] - 1
    cagr = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
    vol = np.std(rets) * np.sqrt(252)  # annualisé
    sharpe = (cagr - rf_mean) / vol if vol > 0 else 0
    dd = (track["net_value"] / track["net_value"].cummax() - 1).min()  # drawdown
    calmar = cagr / abs(dd) if dd != 0 else 0

    rb = compute_rolling_beta(track, benchmark, window=beta_window)
    beta_mean = rb.abs().mean()

    # Sortino
    excess = rets - rf_mean / 252
    down = excess[excess < 0]
    ds = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-8
    sortino = (np.mean(excess) * 252) / ds if ds > 0 else 0

    costs = (
        track["cumulative_costs"].iloc[-1]
        if "cumulative_costs" in track.columns
        else 0
    )

    return {
        "total_return": tr,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": dd,
        "beta_mean_abs": beta_mean,
        "avg_positions": track["n_positions"].mean(),
        "avg_gross": track["gross_exposure"].mean(),
        "total_costs": costs,
    }


def load_best_params(
    json_path: str | Path, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Charge les params gridsearch depuis JSON, sinon defaults."""
    p = Path(json_path)
    if p.exists():
        with open(p) as f:
            raw = json.load(f)
        params = {k: v for k, v in raw.items() if k in defaults}
        return {**defaults, **params}
    else:
        return dict(defaults)


def load_data(data_dir: Path, start: str, end: str) -> dict:
    """Charge les fichiers parquet (prices, PE, universe, benchmark, risk-free)."""
    import time
    t0 = time.time()

    prices = pd.read_parquet(data_dir / "prices.parquet", engine="fastparquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().loc[:end]

    pe_ratios = pd.read_parquet(data_dir / "pe_ratios.parquet", engine="fastparquet")
    pe_ratios.index = pd.to_datetime(pe_ratios.index)
    pe_ratios = pe_ratios.sort_index().loc[:end]

    universe = pd.read_parquet(data_dir / "universe.parquet", engine="fastparquet")
    universe["date"] = pd.to_datetime(universe["date"])

    benchmark = pd.read_parquet(data_dir / "benchmark.parquet", engine="fastparquet")
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark = benchmark.iloc[:, 0]

    rf_df = pd.read_parquet(data_dir / "risk_free.parquet", engine="fastparquet")
    risk_free = rf_df.iloc[:, 0]
    risk_free.index = pd.to_datetime(risk_free.index)
    risk_free = risk_free.sort_index().loc[start:end]

    print(f"Data loaded in {time.time() - t0:.1f}s")
    print(f"  prices:    {prices.shape}")
    print(f"  pe_ratios: {pe_ratios.shape}")
    print(f"  universe:  {len(universe)} rows")
    return {
        "prices": prices,
        "pe_ratios": pe_ratios,
        "universe": universe,
        "benchmark": benchmark,
        "risk_free": risk_free,
    }


def report_strategy(name: str, track: pd.DataFrame, benchmark, risk_free,
                    out_dir: Path):
    """Calcule metrics, affiche resume, sauvegarde CSV."""
    from .indicators import calculate_all_metrics, format_metrics_table
    
    bm_ret = benchmark.pct_change().reindex(track["date"]).fillna(0)
    rf_mean = risk_free.mean() / 100 if risk_free.mean() > 1 else risk_free.mean()

    metrics = calculate_all_metrics(
        track, risk_free_rate=rf_mean, benchmark_returns=bm_ret,
    )

    print(f"\n{'=' * 70}")
    print(f"  {name} -- PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    table = format_metrics_table(metrics)
    print(table.to_string())

    csv = out_dir / f"{name.lower().replace(' ', '_')}_track.csv"
    track.to_csv(csv, index=False)
    print(f"\n  -> Track saved to {csv.name}")

    return metrics


def run_gridsearch(script_name, project_dir, gs_dir):
    import sys, subprocess
    script = gs_dir / script_name
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        cwd=str(project_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} a echoue (code {proc.returncode})")


def extract_ticker_root(ticker):
    parts = ticker.split()
    return parts[0] if len(parts) > 1 else ticker
