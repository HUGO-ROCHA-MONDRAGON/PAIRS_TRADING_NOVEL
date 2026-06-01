"""
portfolio_metrics.py — Métriques additionnelles pour le portefeuille pairs‑trading.

Compléments à ``metrics.py`` : Sortino, skewness, kurtosis, jours min/max,
turnover (eq. 21), rétention/Jaccard (eq. 22), concentration max‑degree.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import annual_return, annual_volatility, max_drawdown, sharpe_ratio


# ───────────────────────────── Rendements ────────────────────────────────
def sortino_ratio(returns: pd.Series, risk_free: pd.Series | None = None,
                  periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    if risk_free is not None:
        r = r.sub(risk_free.reindex(r.index).fillna(0.0), fill_value=0.0)
    downside = r[r < 0]
    if downside.empty:
        return float("nan")
    dd_vol = float(downside.std(ddof=1) * math.sqrt(periods_per_year))
    if dd_vol <= 1e-12:
        return float("nan")
    return annual_return(r, periods_per_year) / dd_vol


def cumulative_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def daily_min(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.min()) if not r.empty else float("nan")


def daily_max(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.max()) if not r.empty else float("nan")


def skewness(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.skew()) if len(r) > 2 else float("nan")


def kurtosis(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.kurtosis()) if len(r) > 3 else float("nan")


def summary_table(
    data: dict[str, pd.Series], risk_free: pd.Series | None = None
) -> pd.DataFrame:
    """Tableau récapitulatif type Tables 2/3 du papier."""
    rows = []
    for name, r in data.items():
        r = r.dropna()
        if r.empty:
            continue
        rows.append({
            "Strategy": name,
            "Cumulative": cumulative_return(r),
            "Annualised": annual_return(r),
            "Volatility": annual_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free),
            "Sortino": sortino_ratio(r, risk_free),
            "MaxDD": max_drawdown(r),
            "Min day": daily_min(r),
            "Max day": daily_max(r),
            "Skew": skewness(r),
            "Kurt": kurtosis(r),
        })
    return pd.DataFrame(rows)


# ────────────────────────── Portefeuille‑centric ─────────────────────────
def selected_pair_assets(selected: pd.DataFrame) -> pd.Series:
    """Pour chaque date de rebal, retourne le set d'actifs utilisés."""
    if selected.empty or "reb_date" not in selected.columns:
        return pd.Series(dtype=object)
    return selected.groupby("reb_date").apply(
        lambda g: set(g["asset_a"]).union(set(g["asset_b"]))
    )


def turnover_series(selected: pd.DataFrame, max_pairs: int = 25) -> pd.Series:
    """
    Turnover ≡ taille du portefeuille / 2N (en %).
    Approxime l'eq. 21 du papier (sans pondération en $).
    """
    assets = selected_pair_assets(selected)
    if assets.empty:
        return pd.Series(dtype=float)
    return pd.Series({d: len(s) / (max_pairs * 2.0) * 100.0 for d, s in assets.items()})


def jaccard_retention(selected: pd.DataFrame) -> pd.Series:
    """
    Rétention mois‑à‑mois (Jaccard sur l'ensemble des actifs sélectionnés).
    Reproduit Figure 4 / eq. 22.
    """
    assets = selected_pair_assets(selected)
    if assets.empty:
        return pd.Series(dtype=float)
    dates = sorted(assets.index)
    out = {}
    for i in range(1, len(dates)):
        prev, curr = assets[dates[i - 1]], assets[dates[i]]
        inter = len(prev & curr)
        union = len(prev | curr) or 1
        out[dates[i]] = inter / union * 100.0
    return pd.Series(out)


def concentration_series(selected: pd.DataFrame) -> pd.Series:
    """
    Concentration ≡ degré max d'un actif / total des apparitions (en %).
    Pour le matching, vaut 1/2N ; pour le baseline, peut être très élevé.
    """
    if selected.empty or "reb_date" not in selected.columns:
        return pd.Series(dtype=float)

    def _conc(g: pd.DataFrame) -> float:
        assets = list(g["asset_a"]) + list(g["asset_b"])
        if not assets:
            return 0.0
        counts = pd.Series(assets).value_counts()
        return float(counts.max() / counts.sum() * 100.0)

    out = selected.groupby("reb_date").apply(_conc)
    out.index = pd.to_datetime(out.index)
    return out


def correlation_matrix(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Reproduit la Table 4 du papier (corrélation inter‑stratégies)."""
    df = pd.concat(
        [s.rename(name) for name, s in returns.items()], axis=1
    ).dropna(how="all")
    return df.corr().round(3)
