"""
theoretical.py — Théorèmes 1‑4 du papier (Table 1).

Hypothèses (Section 4) :
    - Soit ε_t un processus stationnaire de moyenne 0, variance σ².
    - Sur N paires sélectionnées, on ouvre la position quand z = ε/σ franchit
      ±k=2, on cloture au retour à 0.
    - On suppose les paires deux à deux indépendantes (théorème 2).
    - Pour le portefeuille MATCHING, aucun actif n'est partagé entre paires.
    - Pour le portefeuille BASELINE, en moyenne m₂ collisions d'actifs sur
      les N paires.

Quantités calculées (notations du papier) :
    µ₁ : espérance du P&L d'une paire isolée par cycle
    σ₁ : écart‑type du P&L d'une paire isolée
    σ_p² : variance du portefeuille
    Sharpe_M = sqrt(N) · µ₁ / σ₁                 (matching, eq. 14)
    Sharpe_B = sqrt(N) · µ₁ / sqrt(σ₁² + 2 m₂ σ²)  (baseline, eq. 16)

Les valeurs par défaut reproduisent la Table 1 du papier :
    µ₁ = 0.0005, σ₁ = 0.0180, σ = 0.0711, m₂ = 1748, N = 250.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TheoreticalParams:
    mu1: float = 0.0005       # E[PnL d'une paire / cycle]
    sigma1: float = 0.0180    # std[PnL d'une paire / cycle]
    sigma: float = 0.0711     # std du rendement d'un asset
    m2: int = 1748            # nb de paires partageant un actif (baseline)
    N: int = 250              # nb de paires actives


class TheoremCalculator:
    """Calcule les statistiques fermées des Théorèmes 1‑4."""

    def __init__(self, params: TheoreticalParams | None = None) -> None:
        self.p = params or TheoreticalParams()

    # eq. 13‑14 — portefeuille MATCHING (paires indépendantes)
    def matching_sharpe(self) -> float:
        p = self.p
        return math.sqrt(p.N) * p.mu1 / p.sigma1

    def matching_variance(self) -> float:
        return self.p.sigma1 ** 2 / self.p.N

    # eq. 15‑16 — portefeuille BASELINE (m₂ collisions d'actifs)
    def baseline_variance(self) -> float:
        p = self.p
        return (p.sigma1 ** 2 + 2.0 * p.m2 * p.sigma ** 2) / (p.N ** 2)

    def baseline_sharpe(self) -> float:
        # Sharpe = µ_portfolio / σ_portfolio
        #        = µ₁ / sqrt((σ₁² + 2 m₂ σ²) / N²)
        #        = N·µ₁ / sqrt(σ₁² + 2 m₂ σ²)
        p = self.p
        denom = math.sqrt(p.sigma1 ** 2 + 2.0 * p.m2 * p.sigma ** 2)
        return p.N * p.mu1 / denom

    # ------------------------------------------------------------------
    def table1(self) -> pd.DataFrame:
        """Reproduit la Table 1 du papier."""
        p = self.p
        rows = [
            ("µ₁ (mean pair PnL)", f"{p.mu1:.4f}"),
            ("σ₁ (std pair PnL)", f"{p.sigma1:.4f}"),
            ("σ (std asset return)", f"{p.sigma:.4f}"),
            ("m₂ (shared‑asset count)", f"{p.m2}"),
            ("N (pairs)", f"{p.N}"),
            ("σ_p (Matching)", f"{math.sqrt(self.matching_variance()):.6f}"),
            ("σ_p (Baseline)", f"{math.sqrt(self.baseline_variance()):.6f}"),
            ("Sharpe Matching", f"{self.matching_sharpe():.3f}"),
            ("Sharpe Baseline", f"{self.baseline_sharpe():.3f}"),
        ]
        return pd.DataFrame(rows, columns=["Quantity", "Value"])

    # ------------------------------------------------------------------
    def compare_to_paper(self) -> pd.DataFrame:
        """Compare aux valeurs annoncées dans la Table 1 du papier."""
        p = self.p
        paper = {
            "Sharpe Matching": 1.18,
            "Sharpe Baseline": 0.50,
        }
        computed = {
            "Sharpe Matching": self.matching_sharpe(),
            "Sharpe Baseline": self.baseline_sharpe(),
        }
        return pd.DataFrame({
            "Paper": pd.Series(paper),
            "Computed": pd.Series(computed),
            "Δ": pd.Series({k: computed[k] - paper[k] for k in paper}),
        }).round(4)
