from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series | None = None, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    if risk_free is not None:
        r = r.sub(risk_free.reindex(r.index).fillna(0.0), fill_value=0.0)
    vol = annual_volatility(r, periods_per_year)
    if not np.isfinite(vol) or vol <= 1e-12:
        return float("nan")
    return annual_return(r, periods_per_year) / vol


def sharpe_ratio_mean_std(returns: pd.Series, risk_free: pd.Series | None = None, periods_per_year: int = 252) -> float:
    """Sharpe annualisé standard: mean(excess) / std(excess) * sqrt(252)."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    if risk_free is not None:
        r = r.sub(risk_free.reindex(r.index).fillna(0.0), fill_value=0.0)
    std = float(r.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(r.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    equity = (1.0 + r).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def performance_table(data: pd.DataFrame, risk_free: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for col in data.columns:
        r = data[col].dropna()
        if len(r) == 0:
            continue
        rows.append({
            "strategy": col,
            "annual_return": annual_return(r),
            "annual_volatility": annual_volatility(r),
            "sharpe": sharpe_ratio(r, risk_free),
            "max_drawdown": max_drawdown(r),
            "n_days": int(r.count()),
        })
    return pd.DataFrame(rows)
