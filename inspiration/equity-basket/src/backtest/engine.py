"""
Moteur de backtest qui transforme les poids en track de P&L journalier
Gère les coûts de transaction et le suivi des positions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class BacktestEngine:
    """Calcule le P&L journalier à partir d'un calendrier de poids"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        tc_bps: float = 5,  # Coûts de transaction en bps
    ):
        self.initial_capital = initial_capital
        self.tc_bps = tc_bps

    def run(
        self,
        prices: pd.DataFrame,
        weights_df: pd.DataFrame,
        start_date: str = "2001-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Exécute le backtest jour par jour
        Retourne un DataFrame avec le P&L, les coûts, et les expositions
        """
        tc = self.tc_bps / 10_000
        sd, ed = pd.Timestamp(start_date), pd.Timestamp(end_date)
        trading_dates = prices.index[(prices.index >= sd) & (prices.index <= ed)]

        # Préparation du calendrier de poids
        wt_by_date = {}
        for d, grp in weights_df.groupby("date"):
            wt_by_date[d] = grp.set_index("ticker")["weight"]

        # Init des variables
        pv = self.initial_capital
        cum_costs = 0.0
        w_cur = pd.Series(dtype=float)
        rows = []

        # Boucle jour par jour
        for i, date in enumerate(trading_dates):
            is_rebal = date in wt_by_date
            w_new = wt_by_date.get(date, w_cur)

            # Calcul du turnover et des coûts si c'est un rebal
            turnover, tx_cost = 0.0, 0.0
            if is_rebal and len(w_new) > 0:
                all_tk = w_cur.index.union(w_new.index)
                turnover = (
                    w_new.reindex(all_tk, fill_value=0)
                    - w_cur.reindex(all_tk, fill_value=0)
                ).abs().sum()
                tx_cost = turnover * tc * pv
                cum_costs += tx_cost
                w_cur = w_new.copy()

            # Calcul du rendement du portefeuille
            port_ret = 0.0
            if i > 0 and len(w_cur) > 0:
                prev = trading_dates[i - 1]
                rets = (prices.loc[date] / prices.loc[prev]) - 1
                for tk, w in w_cur.items():
                    if tk in rets.index and not pd.isna(rets[tk]):
                        port_ret += w * rets[tk]

            # Mise à jour de la valeur du portefeuille
            pv_new = pv * (1 + port_ret) - tx_cost
            net_ret = (pv_new / pv) - 1 if pv > 0 else 0.0
            pv = pv_new

            # Calcul des expositions long/short
            le = w_cur[w_cur > 0].sum() if len(w_cur) > 0 else 0
            se = w_cur[w_cur < 0].sum() if len(w_cur) > 0 else 0

            rows.append({
                "date": date,
                "portfolio_return": port_ret,
                "portfolio_value": pv,
                "net_value": pv,
                "transaction_cost": tx_cost,
                "cumulative_costs": cum_costs,
                "net_return": net_ret,
                "turnover": turnover,
                "is_rebalance": is_rebal,
                "n_positions": (w_cur != 0).sum() if len(w_cur) > 0 else 0,
                "long_exposure": le,
                "short_exposure": se,
                "gross_exposure": le - se,
                "net_exposure": le + se,
            })

        return pd.DataFrame(rows)


# Fonction wrapper pour compatibilité
def run_multifactor_backtest(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    start_date: str = "2001-01-01",
    end_date: str = "2025-12-31",
    initial_capital: float = 1_000_000,
    tc_bps: float = 10,
) -> pd.DataFrame:
    """Wrapper simple pour lancer le backtest"""
    engine = BacktestEngine(initial_capital=initial_capital, tc_bps=tc_bps)
    return engine.run(prices, weights_df, start_date, end_date)
