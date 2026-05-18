"""
Récupère les données depuis Bloomberg Terminal
Prix, P/E, compo de l'indice, taux sans risque
Sauvegarde en .parquet dans data/raw/ pour éviter de refetch à chaque fois
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import warnings
import traceback

try:
    import blp_bds as blp
    BLP_AVAILABLE = True
except ImportError:
    BLP_AVAILABLE = False
    warnings.warn(
        "blp_bds not available. Make sure blpapi is installed "
        "and Bloomberg Terminal is running."
    )


# Tickers des indices qu'on utilise
INDEX_TICKERS = {
    'SPX': 'SPX Index',
    'MID': 'MID Index',
    'SML': 'SML Index',
    'RTY': 'RTY Index',
    'RIY': 'RIY Index',
}

RISK_FREE_TICKER = 'USGG3M Index'

# Fichiers qu'on s'attend à trouver dans data/raw/
EXPECTED_FILES = [
    'prices.parquet', 'pe_ratios.parquet', 'universe.parquet',
    'risk_free.parquet',
]


class BloombergDataProvider:
    """Classe centrale pour récupérer toutes les data Bloomberg"""

    def __init__(
        self,
        index_name: str = "SPX",
        start_date: str = "2000-01-01",
        end_date: str = "2025-12-31",
        output_path: str | Path | None = "data/raw",
    ):
        self.index_name = index_name
        self.start_date = start_date
        self.end_date = end_date
        self.output_path = Path(output_path) if output_path else None

        self.index_ticker = INDEX_TICKERS.get(index_name)
        if not self.index_ticker:
            raise ValueError(f"Unknown index: {index_name}")

    # Composition de l'indice
    def get_index_members(self, date: str | None = None) -> pd.DataFrame:
        return get_index_members(self.index_ticker, date)

    def get_historical_composition(self, dates: List[str]) -> pd.DataFrame:
        return get_historical_composition(self.index_ticker, dates)

    # Prix et données de marché
    def get_prices(self, tickers: List[str], field: str = "PX_LAST") -> pd.DataFrame:
        return get_prices(tickers, self.start_date, self.end_date, field)

    def get_pe_ratios(self, tickers: List[str]) -> pd.DataFrame:
        return get_prices(tickers, self.start_date, self.end_date, "PE_RATIO")

    def get_risk_free_rate(self, ticker: str = RISK_FREE_TICKER) -> pd.Series:
        return get_risk_free_rate(ticker, self.start_date, self.end_date)

    # Pipeline complet: fetch ou load
    def fetch_all(
        self,
        rebalance_dates: List[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        return fetch_all_data(
            self.index_name, self.start_date, self.end_date,
            rebalance_dates,
            str(self.output_path) if self.output_path else None,
        )

    def load(self) -> Dict[str, pd.DataFrame]:
        if self.output_path is None:
            raise ValueError("output_path not set")
        return load_data(str(self.output_path))

    def fetch_or_load(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        return fetch_or_load(
            self.index_name, self.start_date, self.end_date,
            str(self.output_path) if self.output_path else "data/raw",
            force_refresh,
        )

    def data_exists(self) -> bool:
        if self.output_path is None:
            return False
        return data_exists(str(self.output_path))


# Fonctions utilitaires

def get_index_members(
    index_ticker: str,
    date: Optional[str] = None,
) -> pd.DataFrame:
    """Récup la composition de l'indice à une date donnée"""
    if not BLP_AVAILABLE:
        raise ImportError("blpapi not available")

    try:
        # Appel Bloomberg pour récup les membres de l'indice
        if date:
            dt_clean = date.replace("-", "")
            raw = blp.bds(
                index_ticker,
                "INDX_MWEIGHT_HIST",
                END_DATE_OVERRIDE=dt_clean,
            )
        else:
            raw = blp.bds(index_ticker, "INDX_MEMBERS")

        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame(columns=["ticker"])

        df = raw.copy()

        # Gestion des colonnes multi-index (Bloomberg peut retourner ça)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(str(x) for x in col if str(x) != "").strip()
                for col in df.columns
            ]

        cols_lower = {str(c).lower(): c for c in df.columns}

        # On cherche la colonne qui contient les tickers
        ticker_col = None
        for key in [
            "index_member", "index member",
            "member_ticker_and_exchange_code",
            "member ticker and exchange code",
            "member", "ticker",
        ]:
            if key in cols_lower:
                ticker_col = cols_lower[key]
                break

        if ticker_col is None:
            raise ValueError(
                f"Could not detect ticker column. Columns: {list(df.columns)}"
            )

        # Extraction et nettoyage des tickers
        tickers = df[ticker_col].apply(
            lambda x: x[0] if isinstance(x, (tuple, list)) and len(x) else x
        )
        tickers = tickers.astype(str).str.strip()
        tickers = tickers[
            tickers.notna() & (tickers != "") & (tickers.str.lower() != "nan")
        ]

        return pd.DataFrame({"ticker": tickers.values})

    except Exception as e:
        print(f"Error fetching index members for {index_ticker} @ {date}: {e}")
        return pd.DataFrame(columns=["ticker"])


def get_historical_composition(
    index_ticker: str,
    dates: List[str],
) -> pd.DataFrame:
    """Récup la composition historique mensuelle de l'indice"""
    all_members = []

    for i, date in enumerate(dates):
        members = get_index_members(index_ticker, date)
        n = len(members)
        if n > 0:
            members["date"] = pd.to_datetime(date)
            all_members.append(members[["date", "ticker"]])

    if len(all_members) == 0:
        return pd.DataFrame(columns=["date", "ticker"])

    return pd.concat(all_members, ignore_index=True)


# Helper pour appeler bdh et nettoyer les colonnes
def _fetch_bdh(
    tickers: List[str],
    field: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Appelle Bloomberg bdh puis nettoie les colonnes"""
    if not BLP_AVAILABLE:
        raise ImportError("blpapi not available")

    try:
        data = blp.bdh(
            tickers=tickers,
            flds=[field],
            start_date=start_date,
            end_date=end_date,
        )

        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame()

        # Aplatit le multi-index si nécessaire
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(level=1, axis=1)

        # Nettoyage des noms de colonnes
        data.columns = [
            str(c[0]) if isinstance(c, tuple) else str(c)
            for c in data.columns
        ]
        data.columns = [
            col.replace(' US Equity', '').replace(' Equity', '')
            for col in data.columns
        ]
        return data

    except Exception as e:
        print(f"Erreur bdh ({field}) : {e}")
        traceback.print_exc()
        return pd.DataFrame()


def get_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    field: str = 'PX_LAST',
) -> pd.DataFrame:
    """Récup les données journalières (prix, P/E, etc)"""
    return _fetch_bdh(tickers, field, start_date, end_date)


def get_risk_free_rate(
    ticker: str = 'USGG3M Index',
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31',
) -> pd.Series:
    """Récup le taux sans risque (Treasury Bill 3M)"""
    data = _fetch_bdh([ticker], 'PX_LAST', start_date, end_date)
    if data.empty:
        return pd.Series(dtype=float)
    return data.iloc[:, 0]


def fetch_all_data(
    index_name: str = 'SPX',
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31',
    rebalance_dates: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch complet des data pour le backtest (compo, prix, P/E, rf)"""
    if not BLP_AVAILABLE:
        raise ImportError("blp_bds not available. Install blpapi.")

    index_ticker = INDEX_TICKERS.get(index_name)
    if not index_ticker:
        raise ValueError(f"Unknown index: {index_name}")

    # Composition de l'indice
    if rebalance_dates:
        universe = get_historical_composition(index_ticker, rebalance_dates)
    else:
        members = get_index_members(index_ticker)
        universe = pd.DataFrame({
            'date': pd.to_datetime(end_date),
            'ticker': members['ticker'],
        })

    if len(universe) == 0:
        return {}

    tickers = universe['ticker'].unique().tolist()

    # On ajoute " Equity" pour Bloomberg si pas déjà présent
    tickers_bbg = [
        t if t.endswith(' Equity') else f"{t} Equity"
        for t in tickers
    ]

    # Prix, P/E, Rf
    prices = get_prices(tickers_bbg, start_date, end_date)

    # P/E
    pe_ratios = get_prices(tickers_bbg, start_date, end_date, 'PE_RATIO')

    # Taux sans risque
    risk_free = get_risk_free_rate(RISK_FREE_TICKER, start_date, end_date)

    data = {
        'prices': prices,
        'pe_ratios': pe_ratios,
        'universe': universe,
        'risk_free': risk_free,
        'benchmark': risk_free,
    }

    # Sauvegarde
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(output_dir / 'prices.parquet')
        pe_ratios.to_parquet(output_dir / 'pe_ratios.parquet')
        universe.to_parquet(output_dir / 'universe.parquet')
        risk_free.to_frame('risk_free').to_parquet(output_dir / 'risk_free.parquet')

    return data


def data_exists(output_path: str) -> bool:
    """Vérifie si tous les .parquet sont déjà là"""
    output_dir = Path(output_path)
    return all((output_dir / f).exists() for f in EXPECTED_FILES)


def load_data(output_path: str) -> Dict[str, pd.DataFrame]:
    """Charge les data depuis les .parquet"""
    output_dir = Path(output_path)

    # Lecture des fichiers
    prices    = pd.read_parquet(output_dir / 'prices.parquet', engine='fastparquet')
    pe_ratios = pd.read_parquet(output_dir / 'pe_ratios.parquet', engine='fastparquet')
    universe  = pd.read_parquet(output_dir / 'universe.parquet', engine='fastparquet')
    rf_df     = pd.read_parquet(output_dir / 'risk_free.parquet', engine='fastparquet')

    risk_free = rf_df.iloc[:, 0]

    return {
        'prices': prices,
        'pe_ratios': pe_ratios,
        'universe': universe,
        'risk_free': risk_free,
        'benchmark': risk_free,  # Alias pour compatibilité
    }


def fetch_or_load(
    index_name: str = 'SPX',
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31',
    output_path: str = 'data/raw',
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Charge depuis le disque ou fetch depuis Bloomberg si pas trouvé"""
    if not force_refresh and data_exists(output_path):
        print(f"Données déjà présentes dans {output_path}/ -- chargement depuis le disque.")
        return load_data(output_path)

    print(f"Donnees non trouvees dans {output_path}/ -> fetch depuis Bloomberg...")

    # Génère les dates de rebal mensuelles
    rebalance_dates = pd.date_range(
        start=start_date, end=end_date, freq='ME',
    ).strftime('%Y-%m-%d').tolist()

    return fetch_all_data(
        index_name=index_name,
        start_date=start_date,
        end_date=end_date,
        rebalance_dates=rebalance_dates,
        output_path=output_path,
    )
