from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:  # pragma: no cover
    adfuller = None


def ols_beta_resid(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    xm = x - x.mean()
    denom = float((xm**2).sum())
    if denom < 1e-12:
        return float("nan"), float("nan"), np.empty(0)
    beta = float((xm * (y - y.mean())).sum() / denom)
    alpha = float(y.mean() - beta * x.mean())
    return alpha, beta, y - alpha - beta * x


def adf1_tstat(series: np.ndarray) -> float:
    e = series[np.isfinite(series)]
    if len(e) < 20:
        return float("nan")
    dy = np.diff(e)
    y_lag = e[:-1]
    dy_lag = dy[:-1]
    dy_dep = dy[1:]
    y_lag2 = y_lag[1:]
    n = len(dy_dep)
    if n < 10:
        return float("nan")
    xmat = np.column_stack([np.ones(n), y_lag2, dy_lag])
    try:
        xtx = xmat.T @ xmat
        beta = np.linalg.solve(xtx, xmat.T @ dy_dep)
        resid = dy_dep - xmat @ beta
        sigma2 = (resid @ resid) / (n - 3)
        var_beta = sigma2 * np.linalg.inv(xtx)[1, 1]
        if var_beta <= 0 or not np.isfinite(var_beta):
            return float("nan")
        return float(beta[1] / math.sqrt(var_beta))
    except np.linalg.LinAlgError:
        return float("nan")


def adf1_stats(series: np.ndarray) -> tuple[float, float]:
    """Retourne (t-stat, p-value) pour ADF(1) avec constante.

    Si statsmodels est indisponible, p-value = NaN et t-stat numpy fallback.
    """
    e = series[np.isfinite(series)]
    if len(e) < 20:
        return float("nan"), float("nan")
    if adfuller is None:
        return adf1_tstat(e), float("nan")
    try:
        t_stat, p_value, *_ = adfuller(e, maxlag=1, regression="c", autolag=None)
        return float(t_stat), float(p_value)
    except Exception:
        return adf1_tstat(e), float("nan")


def score_pair(
    lp_a: pd.Series,
    lp_b: pd.Series,
    min_obs: int = 252,
    adf_t_threshold: float | None = None,
    adf_pvalue_threshold: float | None = None,
) -> dict | None:
    frame = pd.concat([lp_a, lp_b], axis=1).dropna()
    if len(frame) < min_obs:
        return None
    first = frame.iloc[:, 0].to_numpy(float)
    second = frame.iloc[:, 1].to_numpy(float)
    best_t = np.inf
    best_p = np.inf
    best = None
    for orientation, y, x in [("a_on_b", first, second), ("b_on_a", second, first)]:
        alpha, beta, resid = ols_beta_resid(y, x)
        if not np.isfinite(alpha):
            continue
        t_stat, p_value = adf1_stats(resid)
        if not np.isfinite(t_stat):
            continue
        # Orientation prioritaire: p-value la plus faible; tie-break: t-stat le plus négatif.
        better = False
        if np.isfinite(p_value):
            better = (p_value < best_p) or (np.isclose(p_value, best_p) and t_stat < best_t)
        else:
            better = (not np.isfinite(best_p)) and (t_stat < best_t)
        if better:
            best_p = p_value
            best_t = t_stat
            best = orientation, alpha, beta, resid, t_stat, p_value
    if best is None:
        return None
    if adf_t_threshold is not None and best[4] >= adf_t_threshold:
        return None
    if adf_pvalue_threshold is not None and np.isfinite(best[5]) and best[5] > adf_pvalue_threshold:
        return None
    orientation, alpha, beta, resid, t_stat, p_value = best
    sigma = float(np.std(resid, ddof=1))
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return None
    return {
        "orientation": orientation,
        "alpha": float(alpha),
        "beta": float(beta),
        "mu": float(resid.mean()),
        "sigma": sigma,
        "adf_t": float(t_stat),
        "adf_pvalue": float(p_value) if np.isfinite(p_value) else float("nan"),
        "weight": float(-t_stat),
    }


def select_matching(edges: pd.DataFrame, max_pairs: int = 25) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    graph = nx.Graph()
    for _, row in edges.iterrows():
        graph.add_edge(row["asset_a"], row["asset_b"], weight=float(row["weight"]))
    matched = nx.max_weight_matching(graph, maxcardinality=False)
    rows = []
    for a, b in matched:
        mask = (((edges["asset_a"] == a) & (edges["asset_b"] == b)) | ((edges["asset_a"] == b) & (edges["asset_b"] == a)))
        found = edges.loc[mask]
        if not found.empty:
            rows.append(found.iloc[0])
    if not rows:
        return pd.DataFrame(columns=edges.columns)
    return pd.DataFrame(rows).sort_values("weight", ascending=False).head(max_pairs).reset_index(drop=True)


def select_baseline(edges: pd.DataFrame, max_pairs: int = 25) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    # Baseline du papier: top paires par plus faible p-value ADF.
    if "adf_pvalue" in edges.columns:
        ranked = edges.sort_values(["adf_pvalue", "adf_t"], ascending=[True, True])
    else:
        ranked = edges.sort_values("weight", ascending=False)
    return ranked.head(max_pairs).reset_index(drop=True)
