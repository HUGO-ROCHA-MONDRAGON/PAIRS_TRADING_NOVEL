"""
bloomberg/
==========
Wrapper Bloomberg — inspiré de xbbg.

Fonctions principales
---------------------
bdh(tickers, flds, start_date, end_date, **overrides)
    Historique de prix/données.
    Retourne MultiIndex (ticker, field) × DatetimeIndex.

bdp(tickers, flds, **overrides)
    Données de référence ponctuelles.
    Retourne DataFrame indexé par security, colonnes en lowercase.

bds(ticker, fld, **overrides)
    Données bulk (membres d'indice, dividendes, etc.).
    Retourne DataFrame plat.

close()
    Ferme la session Bloomberg globale.

Classe haute niveau
-------------------
BloombergDataProvider
    Fetch ou cache (parquet) prix, P/E, composition, taux sans risque.

Usage rapide
------------
    from bloomberg import bdh, bdp, bds

    # Prix AAPL + MSFT sur 2024
    prices = bdh(['AAPL US Equity', 'MSFT US Equity'], 'PX_LAST',
                 '2024-01-01', '2024-12-31')

    # Infos de référence avec override
    ref = bdp('AAPL US Equity', ['Security_Name', 'GICS_Sector_Name'])
    vwap = bdp('AAPL US Equity', 'Eqy_Weighted_Avg_Px', VWAP_Dt='20240115')

    # Membres S&P 500 à une date historique
    members = bds('SPX Index', 'INDX_MWEIGHT_HIST',
                  END_DATE_OVERRIDE='20240101')

    # Pipeline complet (avec cache parquet)
    from bloomberg import BloombergDataProvider
    provider = BloombergDataProvider('SPX', '2000-01-01', '2025-12-31', 'data/raw')
    data = provider.fetch_or_load()
"""
from .blp import bdh, bdp, bds
from ._session import close, get_session
from .provider import BloombergDataProvider, DEFAULT_BENCHMARK_TICKER, DEFAULT_STATIC_FIELDS

__all__ = [
    "bdh",
    "bdp",
    "bds",
    "close",
    "get_session",
    "BloombergDataProvider",
    "DEFAULT_BENCHMARK_TICKER",
    "DEFAULT_STATIC_FIELDS",
]
