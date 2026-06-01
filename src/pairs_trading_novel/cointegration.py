"""
cointegration.py — Analyse de paires (OLS + ADF) en POO.

Wrappe les fonctions numpy rapides définies dans ``core`` dans une classe
réutilisable. La logique de scoring reste identique au papier :

    log(p_j,t) = µ_a + β_a · log(p_i,t) + ε_a,t           (eq. 1)
    ADF(1) sur ε_a → t‑statistique                         (eq. 2)
    poids d'arête w = -t_ADF                              (plus grand = mieux)

On teste les deux orientations (a~b et b~a) et garde la plus stationnaire.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .core import adf1_tstat, ols_beta_resid, score_pair


@dataclass
class PairResult:
    asset_a: str
    asset_b: str
    orientation: str
    alpha: float
    beta: float
    mu: float
    sigma: float
    adf_t: float
    weight: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class PairAnalyzer:
    """
    Calcule les statistiques de cointegration pour un univers de tickers
    sur une fenêtre de formation donnée.

    Paramètres
    ----------
    min_obs            : nb minimal d'observations conjointes valides
    adf_t_threshold    : seuil optionnel sur t‑stat ADF (plus négatif = mieux)
    adf_pvalue_threshold: seuil optionnel sur p-value ADF
    correlation_filter : si > 0, pré‑filtre par |ρ| ≥ seuil (gain de perf)
    max_candidate_pairs: borne optionnelle du nombre de paires scorées par
                         rebalance (top |corr|). None = pas de borne.
    """

    def __init__(
        self,
        min_obs: int = 252,
        adf_t_threshold: float | None = None,
        adf_pvalue_threshold: float | None = None,
        correlation_filter: float = 0.5,
        max_candidate_pairs: int | None = None,
    ) -> None:
        self.min_obs = min_obs
        self.adf_t_threshold = adf_t_threshold
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.correlation_filter = correlation_filter
        self.max_candidate_pairs = max_candidate_pairs

    # ------------------------------------------------------------------
    def candidate_indices(
        self, returns: pd.DataFrame, tickers: list[str]
    ) -> list[tuple[int, int]]:
        n = len(tickers)
        mat = returns[tickers].fillna(0.0).to_numpy(float)
        centered = mat - mat.mean(axis=0)
        scale = np.sqrt((centered ** 2).sum(axis=0))
        scale[scale < 1e-12] = np.nan
        normed = centered / scale
        corr = normed.T @ normed
        i, j = np.triu_indices(n, k=1)
        abs_corr = np.abs(corr[i, j])

        if self.correlation_filter > 0.0:
            keep = abs_corr >= self.correlation_filter
            i = i[keep]
            j = j[keep]
            abs_corr = abs_corr[keep]

        if self.max_candidate_pairs is not None and len(abs_corr) > self.max_candidate_pairs:
            k = int(self.max_candidate_pairs)
            top_idx = np.argpartition(abs_corr, -k)[-k:]
            i = i[top_idx]
            j = j[top_idx]

        return list(zip(i.tolist(), j.tolist()))

    # ------------------------------------------------------------------
    def score_pair(self, lp_a: pd.Series, lp_b: pd.Series) -> dict | None:
        return score_pair(
            lp_a,
            lp_b,
            min_obs=self.min_obs,
            adf_t_threshold=self.adf_t_threshold,
            adf_pvalue_threshold=self.adf_pvalue_threshold,
        )

    # ------------------------------------------------------------------
    def build_edges(
        self,
        log_prices: pd.DataFrame,
        daily_returns: pd.DataFrame,
        tickers: list[str],
        as_of: pd.Timestamp,
        lookback_days: int = 504,
    ) -> pd.DataFrame:
        """
        Construit la table des arêtes pour ``as_of``.
        Retour : DataFrame trié par poids décroissant.
        """
        cols = [
            "asset_a", "asset_b", "orientation", "alpha", "beta",
            "mu", "sigma", "adf_t", "adf_pvalue", "weight",
        ]
        if len(tickers) < 2:
            return pd.DataFrame(columns=cols)
        lp_hist = log_prices[log_prices.index <= as_of].tail(lookback_days)
        ret_hist = daily_returns[daily_returns.index <= as_of].tail(lookback_days)
        pairs = self.candidate_indices(ret_hist, tickers)
        rows = []
        for i, j in pairs:
            a, b = tickers[i], tickers[j]
            scored = self.score_pair(lp_hist[a], lp_hist[b])
            if scored is None:
                continue
            rows.append({"asset_a": a, "asset_b": b, **scored})
        if not rows:
            return pd.DataFrame(columns=cols)
        return (
            pd.DataFrame(rows)
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )
