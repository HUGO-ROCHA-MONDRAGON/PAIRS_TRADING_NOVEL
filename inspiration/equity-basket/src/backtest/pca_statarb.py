"""
Stratégie PCA Statistical Arbitrage.
Mean-reversion sur résidus PCA. Market neutral (dollar + beta) avec vol targeting.
Equal-weight, rebal mensuel, overlay de régime pour réduire l'exposition en marchés volatils.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from base import Strategy
from signals import PCAResidualSignal
from utils import normalize_ticker, universe_at_date_list, compute_stock_betas

# Réexportation rétrocompatible
from utils import compute_rolling_beta  # noqa: F401


class PCAStatArbStrategy(Strategy):
    """Stratégie PCA Stat-Arb. Hérite de Strategy (base.py)."""

    def __init__(self, **kw):
        self.lookback       = kw.get("lookback", 252)
        self.n_components   = kw.get("n_components", 5)
        self.min_history    = kw.get("min_history", 252)
        self.standardise    = kw.get("standardise_vol", True)

        self.zscore_window  = kw.get("zscore_window", 60)
        self.entry_z        = kw.get("entry_zscore", 1.25)
        self.exit_z         = kw.get("exit_zscore", 0.0)
        self.max_hold       = kw.get("max_holding_days", 90)
        self.min_hold       = kw.get("min_holding_days", 42)

        self.max_per_side   = kw.get("max_positions_per_side", 50)

        self.vol_target     = kw.get("vol_target", 0.15)
        self.max_gross      = kw.get("max_gross", 4.0)
        self.tc_bps         = kw.get("tc_bps", 5)

        self.hl_lo          = kw.get("half_life_min", 1)
        self.hl_hi          = kw.get("half_life_max", 30)
        self.rv_mult        = kw.get("max_resid_vol_mult", 4.0)

        self.regime_win     = kw.get("regime_vol_window", 63)
        self.regime_cap     = kw.get("regime_vol_cap", 2.0)

        self.beta_lookback  = kw.get("beta_lookback", 252)
        self.rebal_freq     = kw.get("rebalance_freq", "ME")
        self.weight_smooth  = kw.get("weight_smooth", 0.0)

        # Générateur de signaux PCA déléguant à signals.py
        self._signal = PCAResidualSignal(
            n_components=self.n_components,
            lookback=self.lookback,
            min_history=self.min_history,
            zscore_window=self.zscore_window,
            half_life_min=self.hl_lo,
            half_life_max=self.hl_hi,
            max_resid_vol=self.rv_mult,
            standardise=self.standardise,
        )

    def run(self, prices, universe, benchmark,
            start_date="2001-01-01", end_date="2025-12-31",
            risk_free=None) -> Dict:

        self._print_header()

        # Préparation des données
        rets, bm_ret, bm_vol, bm_vol_med, tdays, rdates, universe = \
            self._prepare_data(prices, universe, benchmark, start_date, end_date)

        # Signaux -> poids bruts
        raw_w = self._generate_raw_weights(prices, universe, rets, bm_ret, rdates)

        # P&L journalier
        track, ccost = self._daily_pnl(
            prices, rets, bm_ret, bm_vol, bm_vol_med, tdays, raw_w
        )

        # DataFrame de poids
        wdf = self._build_weights_df(raw_w, track)

        diag = self._diagnostics(track, rdates, ccost)

        return {"weights": wdf, "track": track,
                "rebalance_dates": sorted(set(d for d, _ in raw_w)),
                "diagnostics": diag}

    def _print_header(self):
        print("PCA Stat-Arb")

    def _prepare_data(self, prices, universe, benchmark, start_date, end_date):
        """Normalise l'univers, calcule les rendements et la vol de régime."""
        universe = universe.copy()
        universe["date"] = pd.to_datetime(universe["date"])
        pcols = set(prices.columns)
        # Normalisation des tickers
        universe["ticker"] = universe["ticker"].apply(lambda t: normalize_ticker(t, pcols))
        universe = universe[universe["ticker"].isin(pcols)].copy()

        rets = prices.pct_change(fill_method=None)
        bm_ret = benchmark.pct_change(fill_method=None)
        # Vol du benchmark pour l'overlay de régime
        bm_vol = bm_ret.rolling(self.regime_win).std() * np.sqrt(252)
        bm_vol_med = bm_vol.expanding().median()

        s0, s1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
        tdays = prices.index[(prices.index >= s0) & (prices.index <= s1)]

        if self.rebal_freq == "D":
            rdates = tdays.tolist()
        else:
            # Snap aux jours de trading réels
            raw = pd.date_range(s0, s1, freq=self.rebal_freq)
            snapped = []
            for d in raw:
                c = tdays[tdays <= d]
                if len(c):
                    snapped.append(c[-1])
            rdates = sorted(set(snapped))

        return rets, bm_ret, bm_vol, bm_vol_med, tdays, rdates, universe

    def _generate_raw_weights(self, prices, universe, rets, bm_ret, rdates):
        """Signaux PCA -> gestion des positions -> poids bruts equal-weight."""
        raw_w = []
        held_long = {}
        held_short = {}

        n_rb = len(rdates)
        pstep = max(1, n_rb // 10)

        for ri, rd in enumerate(rdates):
            tk_z = self._signal.compute_zscores(prices, universe, rd)
            if not tk_z:
                raw_w.append((rd, {}))
                continue

            vtk_set = set(tk_z.keys())

            # Check des positions existantes pour sorties
            keep_long, keep_short = self._check_exits(
                held_long, held_short, tk_z, vtk_set, rd
            )

            # Tri des nouveaux candidats
            held_all = set(keep_long) | set(keep_short)
            long_cands, short_cands = [], []
            for tk, z in tk_z.items():
                if tk in held_all:
                    continue
                if z < -self.entry_z:
                    long_cands.append((tk, z))
                elif z > self.entry_z:
                    short_cands.append((tk, z))

            long_cands.sort(key=lambda x: x[1])
            short_cands.sort(key=lambda x: x[1], reverse=True)

            # Remplir jusqu'à max_per_side
            slots_L = max(0, self.max_per_side - len(keep_long))
            slots_S = max(0, self.max_per_side - len(keep_short))
            new_long = {tk: rd for tk, z in long_cands[:slots_L]}
            new_short = {tk: rd for tk, z in short_cands[:slots_S]}

            held_long = {**keep_long, **new_long}
            held_short = {**keep_short, **new_short}

            longs = list(held_long.keys())
            shorts = list(held_short.keys())

            if not longs and not shorts:
                raw_w.append((rd, {}))
                continue

            # Equal-weight + neutralisations
            w_dict = self._equal_weight_neutralize(
                longs, shorts, prices, rets, bm_ret, rd,
            )
            raw_w.append((rd, w_dict))

        return raw_w

    def _check_exits(self, held_long, held_short, tk_z, vtk_set, rd):
        """Vérifie les positions existantes et supprime celles qui doivent sortir."""
        keep_long, keep_short = {}, {}

        for tk, edt in list(held_long.items()):
            hd = (rd - edt).days
            z = tk_z.get(tk, None)
            if hd >= self.max_hold or tk not in vtk_set or z is None:
                continue
            if hd >= self.min_hold and z > self.exit_z:
                continue
            keep_long[tk] = edt

        for tk, edt in list(held_short.items()):
            hd = (rd - edt).days
            z = tk_z.get(tk, None)
            if hd >= self.max_hold or tk not in vtk_set or z is None:
                continue
            if hd >= self.min_hold and z < -self.exit_z:
                continue
            keep_short[tk] = edt

        return keep_long, keep_short

    def _equal_weight_neutralize(self, longs, shorts, prices, rets, bm_ret, rd):
        """Assignation equal-weight + dollar neutral + beta neutral."""
        nL, nS = len(longs), len(shorts)
        w_dict = {}
        if nL > 0:
            wl = 0.5 / nL
            for tk in longs:
                w_dict[tk] = wl
        if nS > 0:
            ws = 0.5 / nS
            for tk in shorts:
                w_dict[tk] = -ws

        tks = list(w_dict.keys())
        wa = np.array([w_dict[tk] for tk in tks])
        wa -= wa.mean()  # dollar neutral

        # Beta neutral
        ib = prices.index[prices.index <= rd]
        if len(ib) < 2:
            return dict(zip(tks, wa))
        ti = prices.index.get_loc(ib[-1])
        si = max(1, ti - self.lookback)
        rw = rets.iloc[si:ti]
        mr = bm_ret.reindex(rw.index).fillna(0).values

        bv = compute_stock_betas(rw, mr, tks)
        pb = wa @ bv
        bsq = bv @ bv
        if bsq > 1e-12:
            wa -= (pb / bsq) * bv

        # Normalisation gross = 1
        g = np.abs(wa).sum()
        if g > 0:
            wa /= g

        return dict(zip(tks, wa))

    def _daily_pnl(self, prices, rets, bm_ret, bm_vol, bm_vol_med, tdays, raw_w):
        """P&L journalier + ciblage de volatilité + overlay de régime."""
        rows = []
        pv = 1_000_000.0
        cw = {}
        ccost = 0.0
        vs = 1.0

        rmap = {d: w for d, w in raw_w}
        rset = set(rmap.keys())
        evar = 0.0
        elam = 0.94

        for i, dt in enumerate(tdays):
            is_rb = dt in rset

            if is_rb:
                rwd = rmap[dt]
                if rwd:
                    # Vol targeting
                    if i >= 40 and evar > 1e-12:
                        rv = np.sqrt(evar * 252)
                        vs = np.clip(self.vol_target / rv, 0.3, self.max_gross)
                    else:
                        vs = 1.0
                    # Overlay de régime
                    if dt in bm_vol.index and dt in bm_vol_med.index:
                        cv = bm_vol.loc[dt]
                        mv2 = bm_vol_med.loc[dt]
                        if pd.notna(cv) and pd.notna(mv2) and mv2 > 0:
                            vr = cv / mv2
                            if vr > self.regime_cap:
                                vs *= self.regime_cap / vr
                    tw = {tk: w * vs for tk, w in rwd.items()}
                    # Lissage des poids
                    if self.weight_smooth > 0 and cw:
                        a = self.weight_smooth
                        all_tks = set(list(tw.keys()) + list(cw.keys()))
                        tw = {tk: (1 - a) * tw.get(tk, 0) + a * cw.get(tk, 0)
                              for tk in all_tks}
                        tw = {tk: w for tk, w in tw.items() if abs(w) > 1e-9}
                    g = sum(abs(v) for v in tw.values())
                    if g > self.max_gross:
                        tw = {tk: w * self.max_gross / g for tk, w in tw.items()}
                    nw = tw
                else:
                    nw = {}
            else:
                nw = cw

            to = 0.0
            tc = 0.0
            if is_rb:
                atk = set(list(cw.keys()) + list(nw.keys()))
                for tk in atk:
                    to += abs(nw.get(tk, 0) - cw.get(tk, 0))
                tc = to * (self.tc_bps / 1e4) * pv
                ccost += tc
                cw = dict(nw)

            if i == 0:
                pr = 0.0
                pv -= tc
            else:
                pd2 = tdays[i - 1]
                if cw and pd2 in prices.index and dt in prices.index:
                    dr = (prices.loc[dt] / prices.loc[pd2]) - 1
                    pr = sum(cw.get(tk, 0) * dr.get(tk, 0)
                             for tk in cw
                             if tk in dr.index and np.isfinite(dr.get(tk, 0)))
                else:
                    pr = 0.0
                pv = pv * (1 + pr) - tc

            nr = (pv / rows[-1]["portfolio_value"] - 1) if i > 0 and rows[-1]["portfolio_value"] > 0 else 0.0
            evar = elam * evar + (1 - elam) * nr ** 2

            le = sum(w for w in cw.values() if w > 0)
            se = sum(w for w in cw.values() if w < 0)
            rows.append({
                "date": dt, "portfolio_return": pr,
                "portfolio_value": pv, "net_value": pv,
                "transaction_cost": tc, "cumulative_costs": ccost,
                "net_return": nr, "turnover": to,
                "is_rebalance": is_rb,
                "n_positions": sum(1 for w in cw.values() if abs(w) > 1e-9),
                "long_exposure": le, "short_exposure": se,
                "gross_exposure": le + abs(se), "net_exposure": le + se,
                "vol_scale": vs,
            })

        track = pd.DataFrame(rows)
        return track, ccost

    def _build_weights_df(self, raw_w, track):
        """Assemble le DataFrame de poids pour l'export."""
        dvs = {}
        for _, r in track[track["is_rebalance"]].iterrows():
            dvs[r["date"]] = r["vol_scale"]
        wrecs = []
        for d, wd in raw_w:
            if not wd:
                continue
            v = dvs.get(d, 1.0)
            for tk, w in wd.items():
                if abs(w) < 1e-9:
                    continue
                wrecs.append({"date": d, "ticker": tk, "weight": w * v,
                              "side": "LONG" if w > 0 else "SHORT"})
        wdf = pd.DataFrame(wrecs)
        if not wdf.empty:
            wdf.sort_values(["date", "ticker"], inplace=True)
            wdf.reset_index(drop=True, inplace=True)
        return wdf

    @staticmethod
    def _diagnostics(track, rdates, ccost):
        return {
            "n_rebal": len(rdates),
            "avg_positions": track["n_positions"].mean(),
            "avg_gross": track["gross_exposure"].mean(),
            "avg_turnover": (track.loc[track["is_rebalance"], "turnover"].mean()
                             if track["is_rebalance"].any() else 0),
        }


def subperiod_metrics(track, risk_free_rate=0.02):
    """Tableau de rendement/risque par sous-période."""
    periods = {
        "2001-2007": ("2001-01-01", "2007-12-31"),
        "2008-2009": ("2008-01-01", "2009-12-31"),
        "2010-2019": ("2010-01-01", "2019-12-31"),
        "2020-2025": ("2020-01-01", "2025-12-31"),
        "Full": (track["date"].min(), track["date"].max()),
    }
    out = []
    for lab, (s, e) in periods.items():
        sub = track[(track["date"] >= pd.Timestamp(s)) & (track["date"] <= pd.Timestamp(e))]
        if len(sub) < 20:
            continue
        r = sub["net_return"].values
        nd = len(r)
        ny = nd / 252
        tr = sub["net_value"].iloc[-1] / sub["net_value"].iloc[0] - 1
        cagr = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
        vol = np.std(r) * np.sqrt(252)
        # Sharpe ratio avec méthode standard (cohérent avec calculate_sharpe_ratio)
        rf_daily = risk_free_rate / 252
        excess_returns = r - rf_daily
        sr = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
        dd = (sub["net_value"] / sub["net_value"].cummax() - 1).min()
        to = sub.loc[sub["is_rebalance"], "turnover"].mean() if sub["is_rebalance"].any() else 0
        out.append({"Period": lab, "CAGR": f"{cagr:.2%}", "Vol": f"{vol:.2%}",
                     "Sharpe": f"{sr:.2f}", "MaxDD": f"{dd:.2%}",
                     "TO/rebal": f"{to:.1%}",
                     "Gross": f"{sub['gross_exposure'].mean():.2f}",
                     "Net": f"{sub['net_exposure'].mean():.4f}", "Days": nd})
    return pd.DataFrame(out)
