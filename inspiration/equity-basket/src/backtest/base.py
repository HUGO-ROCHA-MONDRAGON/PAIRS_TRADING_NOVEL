"""
Classes de base pour les signaux et stratégies
Définit les interfaces que toutes les strats doivent implémenter

- SignalGenerator: pour les générateurs de signaux (PCA, Value, etc)
- Strategy: pour les stratégies complètes (run le backtest)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


# Classe de base pour tous les générateurs de signaux
# Permet de garantir que PCA et Value ont la même structure
class SignalGenerator(ABC):
    """Interface pour les générateurs de signaux (PCA, Value, etc)
    Tous doivent implémenter generate_signals()
    """

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        universe: Optional[pd.DataFrame] = None,
        **kwargs,  # Pour passer des trucs supplémentaires genre pe_ratios
    ) -> pd.DataFrame:
        """Génère les signaux de trading pour chaque date de rebal
        Doit retourner au minimum: [date, ticker] + colonnes de signaux
        """
        ...

# Classe de base pour toutes les stratégies
# PCA et Value héritent de ça
class Strategy(ABC):
    """Interface pour les stratégies complètes
    Gère la génération de signaux + allocation + backtest
    """

    @abstractmethod
    def run(
        self,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        benchmark: pd.Series,
        start_date: str = "2001-01-01", 
        end_date: str = "2025-12-31",
        risk_free: Optional[pd.Series] = None,
    ) -> Dict:
        """Lance le backtest complet de la strat
        Retour: dict avec 'track' (P&L journalier) et 'weights' (poids par date)
        """
        ...
