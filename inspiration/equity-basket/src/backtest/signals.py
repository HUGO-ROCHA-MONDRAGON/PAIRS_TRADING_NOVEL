"""
Génération de signaux pour les stratégies long-short equity.

PCAResidualSignal: Mean-reversion sur résidus PCA. Z-scores sur résidus cumulés avec filtres half-life.
ValueSignalGenerator: Classement par P/E trailing en quintiles. P/E bas -> LONG, P/E élevé -> SHORT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from sklearn.decomposition import PCA

from base import SignalGenerator
from utils import (
    normalize_ticker,
    force_date_column,
    constituents_to_long,
    normalize_universe_tickers,
    universe_at_date,
    universe_at_date_list,
    get_rebalance_dates,          # re-export for backward compat
)


def _hl_ar1(cum: np.ndarray) -> float:
    """Half-life d'un AR(1) sur le résidu cumulé. Retourne inf si pas mean-reverting."""
    if len(cum) < 20:
        return np.inf
    X, Y = cum[:-1], cum[1:]
    n = len(X)
    Sx, Sy, Sxx, Sxy = X.sum(), Y.sum(), (X * X).sum(), (X * Y).sum()
    den = n * Sxx - Sx ** 2
    if abs(den) < 1e-15:
        return np.inf
    b = (n * Sxy - Sx * Sy) / den
    if b <= 0 or b >= 1:
        return np.inf
    return np.log(2) / (-np.log(b))


class PCAResidualSignal(SignalGenerator):
    """
    Générateur de signaux PCA Stat-Arb.
    PCA sur rendements -> résidus -> z-scores avec filtres half-life et vol résiduelle.
    """

    def __init__(self, **kw):
        self.n_components  = kw.get("n_components", 5)
        self.lookback      = kw.get("lookback", 252)
        self.min_history   = kw.get("min_history", 252)
        self.zscore_window = kw.get("zscore_window", 60)
        self.hl_lo         = kw.get("half_life_min", 1)
        self.hl_hi         = kw.get("half_life_max", 30)
        self.rv_mult       = kw.get("max_resid_vol", 4.0)
        self.standardise   = kw.get("standardise", True)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        universe: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Calcule les z-scores à chaque date de rebalancement et retourne un DataFrame."""
        rows = []
        for d in pd.to_datetime(rebalance_dates):
            zscores = self.compute_zscores(prices, universe, d)
            for tk, z in zscores.items():
                rows.append({"date": d, "ticker": tk, "z_score": z})
        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "z_score"])
        return pd.DataFrame(rows)

    def compute_zscores(
        self,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Dict[str, float]:
        """Calcule les z-scores pour toutes les actions éligibles à la date donnée."""
        pcols = set(prices.columns)
        u = universe.copy()
        u["date"] = pd.to_datetime(u["date"])
        u["ticker"] = u["ticker"].apply(lambda t: normalize_ticker(t, pcols))
        utk = universe_at_date_list(u, date)
        # Besoin d'assez d'actions pour la PCA
        if len(utk) < self.n_components + 30:
            return {}

        rets = prices.pct_change(fill_method=None)
        ib = prices.index[prices.index <= date]
        if len(ib) < self.lookback + 1:
            return {}
        ti = prices.index.get_loc(ib[-1])
        si = max(1, ti - self.lookback)
        rw = rets.iloc[si:ti]

        # Filtrage des tickers: historique suffisant + pas trop de NaN

        vtk = [
            tk for tk in utk
            if tk in rets.columns
            and rets[tk].iloc[:ti].notna().sum() >= self.min_history
            and rw[tk].notna().sum() >= rw.shape[0] * 0.7
        ]
        if len(vtk) < self.n_components + 30:
            return {}

        R = rw[vtk].fillna(0.0)
        Rm, Rs = R.mean(), R.std().replace(0, 1e-8)
        Rn = (R - Rm) / Rs if self.standardise else (R - Rm)

        # PCA

        K = min(self.n_components, len(vtk) - 1, Rn.shape[0] - 1)
        if K < 1:
            return {}

        pca = PCA(n_components=K, random_state=123)
        F = pca.fit_transform(Rn.values)
        L = pca.components_.T
        resid = Rn.values - F @ L.T

        # Z-score sur résidus cumulés

        zw = min(self.zscore_window, resid.shape[0])
        rt = resid[-zw:]
        cum = rt.cumsum(axis=0)

        rvols = np.std(rt, axis=0)
        med_rv = np.median(rvols[rvols > 0]) if np.any(rvols > 0) else 1.0

        cum_mean = cum.mean(axis=0)
        cum_std = cum.std(axis=0)
        cum_std[cum_std < 1e-10] = 1e-10

        # Filtrage et calcul final des z-scores
        result = {}
        for i, tk in enumerate(vtk):
            if rvols[i] > self.rv_mult * med_rv:
                continue
            hl = _hl_ar1(cum[:, i])
            if hl < self.hl_lo or hl > self.hl_hi:
                continue
            z = (cum[-1, i] - cum_mean[i]) / cum_std[i]
            if np.isfinite(z):
                result[tk] = float(z)
        return result


class ValueSignalGenerator(SignalGenerator):
    """
    Signal value basé sur le ratio P/E trailing.
    P/E bas -> LONG, P/E élevé -> SHORT. Bucketing par quintiles avec buffer d'hystérésis optionnel.
    """

    def __init__(
        self,
        n_buckets: int = 5,
        long_buckets: Optional[List[int]] = None,
        short_buckets: Optional[List[int]] = None,
        buffer: int = 0,
    ):
        self.n_buckets = int(n_buckets)
        self.buffer = int(buffer)
        self.long_buckets = self._parse_buckets(long_buckets) if long_buckets is not None else [1]
        self.short_buckets = self._parse_buckets(short_buckets) if short_buckets is not None else [self.n_buckets]

    @staticmethod
    def _parse_buckets(val):
        """Parse buckets depuis JSON si besoin."""
        import json as _json
        if isinstance(val, str):
            val = _json.loads(val)
        if isinstance(val, (int, float)):
            return [int(val)]
        return [int(x) for x in val]

    def generate_signals(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        universe: Optional[pd.DataFrame] = None,
        *,
        pe_ratios: Optional[pd.DataFrame] = None,
        use_low_vol: bool = True,
        vol_window: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Pipeline: extraction P/E -> bucketing -> assignation sides.
        Overlay low-vol optionnel + filtre univers point-in-time.
        """
        if pe_ratios is None:
            raise ValueError("pe_ratios requis pour ValueSignalGenerator")

        rebalance_dates = pd.to_datetime(rebalance_dates)

        # Signal value
        value_df = self.calculate_value_signal(pe_ratios, rebalance_dates)
        signals = value_df.copy()

        # Signal low-vol (optionnel)
        if use_low_vol:
            vol_df = self.calculate_low_vol_signal(prices, rebalance_dates, vol_window)
            signals = pd.merge(signals, vol_df, on=["date", "ticker"], how="outer")
            signals = force_date_column(signals)

        # Filtre univers (sans biais de survivant)
        if universe is not None:
            u_long = constituents_to_long(universe)
            u_long = force_date_column(u_long)
            u_long["ticker"] = u_long["ticker"].astype(str).str.strip()
            u_long = normalize_universe_tickers(u_long, prices.columns)

            rows = []
            for d in rebalance_dates:
                tks = universe_at_date(u_long, d)
                if not tks.empty:
                    rows.append(pd.DataFrame({"date": pd.Timestamp(d), "ticker": tks.values}))
            if rows:
                u_rebal = pd.concat(rows, ignore_index=True)
                u_rebal["ticker"] = u_rebal["ticker"].astype(str).str.strip()
                signals = pd.merge(signals, u_rebal, on=["date", "ticker"], how="inner")
                signals = force_date_column(signals)
            else:
                signals = signals.iloc[0:0].copy()

        # Bucketing (P/E bas -> bucket 1)
        signals = self.assign_buckets(signals, "pe_value", "pe_bucket", ascending=True)
        signals = self.assign_sides(signals, "pe_bucket", "side_value")

        # Buffer d'hystérésis
        if self.buffer > 0:
            signals = self._apply_buffer(signals, "pe_bucket", "side_value")

        if use_low_vol:
            signals = self.assign_buckets(signals, "vol_score", "vol_bucket", ascending=True)
            signals = self.assign_sides(signals, "vol_bucket", "side_vol")

        signals = signals.sort_values(["date", "ticker"]).reset_index(drop=True)
        return signals

    def calculate_value_signal(
        self, pe_ratios: pd.DataFrame, rebalance_dates: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """Extrait le P/E trailing à chaque date de rebalancement."""
        rows = []
        for d in pd.to_datetime(rebalance_dates):
            idx = pe_ratios.index[pe_ratios.index <= d]
            if len(idx) == 0:
                continue
            pe = pe_ratios.loc[idx[-1]].dropna()
            pe = pe[pe > 0]
            for tk, v in pe.items():
                rows.append({"date": d, "ticker": str(tk), "pe_value": float(v)})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "ticker", "pe_value"])
        return force_date_column(df)

    def calculate_low_vol_signal(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        vol_window: int = 60,
    ) -> pd.DataFrame:
        """Volatilité réalisée glissante à chaque date de rebalancement."""
        rows = []
        for d in pd.to_datetime(rebalance_dates):
            dates_before = prices.index[prices.index <= d]
            if len(dates_before) < vol_window:
                continue
            loc = prices.index.get_loc(dates_before[-1])
            pw = prices.iloc[max(0, loc - vol_window) : loc + 1]
            vol = pw.pct_change(fill_method=None).dropna().std() * np.sqrt(252)
            vol = vol.dropna()
            vol = vol[vol > 0]
            for tk, v in vol.items():
                rows.append({"date": d, "ticker": str(tk), "vol_score": float(v)})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "ticker", "vol_score"])
        return force_date_column(df)

    def assign_buckets(
        self,
        signals: pd.DataFrame,
        signal_col: str,
        bucket_col: str,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Bucketing par quantiles par date. ascending=True -> valeur basse = bucket 1."""
        if signals is None or signals.empty:
            return pd.DataFrame(columns=["date", "ticker", signal_col, bucket_col])

        signals = force_date_column(signals)
        out = signals.copy()

        def _per_date(g):
            valid = g[signal_col].notna()
            vg = g.loc[valid].copy()
            if vg.empty:
                g[bucket_col] = np.nan
                return g
            try:
                b = pd.qcut(vg[signal_col], q=self.n_buckets, labels=False, duplicates="drop") + 1
            except Exception:
                r = vg[signal_col].rank(pct=True)
                b = pd.cut(r, bins=self.n_buckets, labels=False, include_lowest=True) + 1
            if not ascending:
                b = self.n_buckets - b + 1
            g[bucket_col] = np.nan
            g.loc[valid, bucket_col] = b.astype("Int64")
            return g

        out = out.groupby("date", group_keys=False).apply(_per_date)
        out = force_date_column(out)
        out[bucket_col] = out[bucket_col].astype("Int64")
        return out

    def assign_sides(
        self, signals: pd.DataFrame, bucket_col: str, side_col: str
    ) -> pd.DataFrame:
        """Associe les numéros de bucket à LONG / SHORT / NEUTRAL."""
        def _side(b):
            if pd.isna(b):
                return "NEUTRAL"
            if b in self.long_buckets:
                return "LONG"
            if b in self.short_buckets:
                return "SHORT"
            return "NEUTRAL"
        signals[side_col] = signals[bucket_col].apply(_side)
        return signals

    def _apply_buffer(
        self, signals: pd.DataFrame, bucket_col: str, side_col: str
    ) -> pd.DataFrame:
        """Buffer d'hystérésis pour éviter trop de turnover. Conserve position si bucket pas trop dérivé."""
        buf = self.buffer
        long_max = max(self.long_buckets) + buf
        short_min = min(self.short_buckets) - buf

        dates = sorted(signals["date"].unique())
        prev_sides: dict = {}

        chunks = []
        for d in dates:
            mask = signals["date"] == d
            df_d = signals.loc[mask].copy()

            for idx, row in df_d.iterrows():
                tk = row["ticker"]
                bucket = row[bucket_col]
                new_side = row[side_col]
                prev = prev_sides.get(tk, "NEUTRAL")

                if pd.isna(bucket):
                    df_d.at[idx, side_col] = "NEUTRAL"
                    prev_sides[tk] = "NEUTRAL"
                    continue

                b = int(bucket)
                if prev == "LONG" and new_side == "NEUTRAL":
                    if b <= long_max:
                        df_d.at[idx, side_col] = "LONG"
                        prev_sides[tk] = "LONG"
                    else:
                        prev_sides[tk] = "NEUTRAL"
                elif prev == "SHORT" and new_side == "NEUTRAL":
                    if b >= short_min:
                        df_d.at[idx, side_col] = "SHORT"
                        prev_sides[tk] = "SHORT"
                    else:
                        prev_sides[tk] = "NEUTRAL"
                else:
                    prev_sides[tk] = new_side

            chunks.append(df_d)

        return pd.concat(chunks, ignore_index=True)



