"""
signals.py — Z‑score et Q‑score pour le trading de paires.

Z‑Score (continu, eq. 3 du papier) :
    z_t = (ε_t − µ) / σ
    Position long si z ≤ −k, short si z ≥ +k, exit sur retour à |z| ≤ exit
    ou stop |z| ≥ stop.

Q‑Score (eq. 19‑20) :
    q_t = (ε_t − τ50) / (τ75 − τ25)
    S_t^q = sign(q_t) · round(|q_t|)
    où τ_p sont les percentiles empiriques estimés sur la fenêtre de formation.
    S_t^q remplace directement la position (long si > 0, short si < 0).
    Winsorisation à ±3 pour limiter l'effet des outliers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpreadSpec:
    """Spécifie une paire et ses paramètres de formation (statiques)."""

    asset_y: str            # ticker dépendant (y dans l'OLS)
    asset_x: str            # ticker explicatif
    alpha: float
    beta: float
    mu: float               # moyenne du résidu sur la formation
    sigma: float            # std du résidu sur la formation
    tau25: float | None = None
    tau50: float | None = None
    tau75: float | None = None


def spread_series(spec: SpreadSpec, log_prices: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Spread = y − (α + β·x) projeté sur ``index``."""
    lp = log_prices.reindex(index)
    return lp[spec.asset_y] - (spec.alpha + spec.beta * lp[spec.asset_x])


class TradingSignal(ABC):
    @abstractmethod
    def positions(self, spec: SpreadSpec, log_prices: pd.DataFrame,
                  index: pd.DatetimeIndex) -> pd.Series: ...


# ---------------------------------------------------------------- Z‑Score
class ZScoreSignal(TradingSignal):
    """
    Z‑score "continu" : la position est −1 / 0 / +1 selon les bandes.
    Compatible avec la convention paper k=2 (entry).
    """

    def __init__(self, entry: float = 2.0, exit: float = 0.5,
                 stop: float = 4.0, winsor: float = 3.0) -> None:
        self.entry = entry
        self.exit = exit
        self.stop = stop
        self.winsor = winsor

    def zscore(self, spec: SpreadSpec, log_prices: pd.DataFrame,
               index: pd.DatetimeIndex) -> pd.Series:
        sp = spread_series(spec, log_prices, index)
        z = (sp - spec.mu) / spec.sigma
        return z.clip(-self.winsor, self.winsor)

    def positions(self, spec: SpreadSpec, log_prices: pd.DataFrame,
                  index: pd.DatetimeIndex) -> pd.Series:
        z = self.zscore(spec, log_prices, index).to_numpy(float)
        pos = np.zeros(len(z))
        cur = 0.0
        for i, v in enumerate(z):
            if not np.isfinite(v):
                cur = 0.0
            elif cur == 0.0:
                if v >= self.entry:
                    cur = -1.0
                elif v <= -self.entry:
                    cur = 1.0
            else:
                if abs(v) <= self.exit or abs(v) >= self.stop:
                    cur = 0.0
            pos[i] = cur
        return pd.Series(pos, index=index, name="position_z")


# ---------------------------------------------------------------- Q‑Score
class QScoreSignal(TradingSignal):
    """
    Q‑score (eq. 19‑20) : position entière proportionnelle à l'écart
    inter‑quartile par rapport à la médiane.
        S_q = sign(q) · round(|q|),  q = (ε − τ50) / (τ75 − τ25)
    On winsorise à ±round_max pour limiter le levier.
    """

    def __init__(self, round_max: int = 3) -> None:
        self.round_max = round_max

    def positions(self, spec: SpreadSpec, log_prices: pd.DataFrame,
                  index: pd.DatetimeIndex) -> pd.Series:
        if spec.tau25 is None or spec.tau75 is None or spec.tau50 is None:
            raise ValueError("Q‑score requires tau25/tau50/tau75 on SpreadSpec.")
        iqr = max(spec.tau75 - spec.tau25, 1e-10)
        sp = spread_series(spec, log_prices, index)
        q = (sp - spec.tau50) / iqr
        # sens contraire : on parie sur le retour à la médiane
        pos = -np.sign(q) * np.minimum(np.round(np.abs(q)), self.round_max)
        pos = pos.where(np.isfinite(pos), 0.0)
        return pos.rename("position_q")


# ----------------------------------------------------- conversion edges → spec
def edge_to_spec(row: pd.Series, *, with_quantiles: pd.DataFrame | None = None) -> SpreadSpec:
    """Convertit une ligne d'``edges`` (cf. cointegration) en SpreadSpec."""
    if row["orientation"] == "a_on_b":
        ya, xa = row["asset_a"], row["asset_b"]
    else:
        ya, xa = row["asset_b"], row["asset_a"]
    tau25 = tau50 = tau75 = None
    if with_quantiles is not None and (ya, xa) in with_quantiles.index:
        q = with_quantiles.loc[(ya, xa)]
        tau25, tau50, tau75 = float(q["q25"]), float(q["q50"]), float(q["q75"])
    return SpreadSpec(
        asset_y=ya, asset_x=xa,
        alpha=float(row["alpha"]), beta=float(row["beta"]),
        mu=float(row["mu"]), sigma=float(row["sigma"]),
        tau25=tau25, tau50=tau50, tau75=tau75,
    )
