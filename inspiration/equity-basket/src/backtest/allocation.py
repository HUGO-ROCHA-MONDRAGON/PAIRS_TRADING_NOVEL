"""
Allocation des poids de portefeuille
Transforme les signaux en weights equal-weight market-neutral
Gère aussi la neutralisation du beta
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List
from utils import compute_stock_betas 


class AllocationEngine:
    """Convertit les signaux en poids equal-weight market-neutral"""

    def __init__(
        self,
        method: str = "equal_weight",
        vol_window: int = 60,
        long_target: float = 1.0,
        short_target: float = -1.0,
    ):
        self.method = method
        self.vol_window = vol_window
        self.long_target = long_target
        self.short_target = short_target

    def equal_weight_allocation(
        self, tickers_long: List[str], tickers_short: List[str]
    ) -> pd.Series:
        """Poids égaux pour chaque side"""
        weights = pd.Series(dtype=float)

        # Répartition uniforme côté long par ex, si 10 actions = 0.1 chaque action
        if len(tickers_long) > 0:
            w = self.long_target / len(tickers_long)
            for tk in tickers_long:
                weights[tk] = w

        # Pareil pour les shorts
        if len(tickers_short) > 0:
            w = self.short_target / len(tickers_short)
            for tk in tickers_short:
                weights[tk] = w

        return weights

    def allocate_for_date(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        date: pd.Timestamp,
        signal_type: str = "mom",
    ) -> pd.DataFrame:
        """Poids pour une date de rebal donnée"""
        sig = signals[signals["date"] == date].copy()
        if sig.empty:
            return pd.DataFrame(columns=["date", "ticker", "weight", "side"])

        # On cherche la colonne qui correspond au type de signal (mom ou value)
        side_col = f"side_{signal_type}"
        if side_col not in sig.columns:
            raise ValueError(f"Column {side_col} not found in signals")

        # Récup des tickers à acheter et shorter
        tickers_long = sig.loc[sig[side_col] == "LONG", "ticker"].tolist()
        tickers_short = sig.loc[sig[side_col] == "SHORT", "ticker"].tolist()

        if not tickers_long and not tickers_short:
            return pd.DataFrame(columns=["date", "ticker", "weight", "side"])

        # Calcul des poids equal-weight
        weights = self.equal_weight_allocation(tickers_long, tickers_short)

        rows = [
            {"date": date, "ticker": tk, "weight": w, "side": "LONG" if w > 0 else "SHORT"}
            for tk, w in weights.items()
        ]
        return pd.DataFrame(rows)

    def generate_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        signal_type: str = "mom",
    ) -> pd.DataFrame:
        """Génère tous les poids pour les dates de rebal"""
        parts = []
        # Boucle sur toutes les dates de rebalancement
        for d in rebalance_dates:
            w = self.allocate_for_date(signals, prices, d, signal_type)
            if not w.empty:
                parts.append(w)

        if not parts:
            return pd.DataFrame(columns=["date", "ticker", "weight", "side"])

        df = pd.concat(parts, ignore_index=True)
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)

    def beta_neutralize(
        self,
        weights_df: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark: pd.Series,
        lookback: int = 252,
    ) -> pd.DataFrame:
        """Neutralise le beta par projection orthogonale
        Etapes: dollar neutral, calcul beta, projection, renorm à 1.0
        """
        # Calcul des returns du benchmark et des actions
        bm_ret = benchmark.pct_change(fill_method=None).dropna()
        stk_ret = prices.pct_change(fill_method=None)
        new_rows = []

        # On traite chaque date de rebal séparément
        for date in sorted(weights_df["date"].unique()):
            grp = weights_df[weights_df["date"] == date]
            tickers = grp["ticker"].values
            w = grp["weight"].values.astype(float).copy()

            # Vérif qu'on a assez d'historique pour calculer le beta
            idx = stk_ret.index[stk_ret.index <= pd.Timestamp(date)]
            if len(idx) < lookback:
                # Pas assez de data, on garde les poids tels quels
                for tk, wi in zip(tickers, w):
                    new_rows.append({"date": date, "ticker": tk, "weight": wi,
                                     "side": "LONG" if wi > 0 else "SHORT"})
                continue

            # Extraction de la fenêtre de lookback
            window = stk_ret.loc[idx[-lookback:]]
            bm_window = bm_ret.reindex(window.index).fillna(0).values
            bm_var = np.var(bm_window)

            # Si le benchmark bouge pas, la projection sert à rien
            if bm_var < 1e-12:
                for tk, wi in zip(tickers, w):
                    new_rows.append({"date": date, "ticker": tk, "weight": wi,
                                     "side": "LONG" if wi > 0 else "SHORT"})
                continue

            # Premier truc: on centre les poids (mean = 0)
            w -= w.mean()

            # On récup les betas de chaque action vs le bench
            betas = compute_stock_betas(window, bm_window, list(tickers))

            # Projection orthogonale: on retire la composante beta
            # Comme ça le ptf final a un beta proche de 0
            port_beta = w @ betas
            beta_sq = betas @ betas
            if beta_sq > 1e-12:
                w -= (port_beta / beta_sq) * betas

            # Renormalisation pour que l'expo brute = 1.0
            gross = np.abs(w).sum()
            if gross > 0:
                w /= gross

            for tk, wi in zip(tickers, w):
                new_rows.append({"date": date, "ticker": tk, "weight": wi,
                                 "side": "LONG" if wi > 0 else "SHORT"})

        return pd.DataFrame(new_rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def beta_neutralize_weights(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    lookback: int = 252,
) -> pd.DataFrame:
    """Wrapper pour neutralisation beta"""
    # Fonction helper pour appeler facilement depuis d'autres modules
    engine = AllocationEngine()
    return engine.beta_neutralize(weights_df, prices, benchmark, lookback)
