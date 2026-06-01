"""
bloomberg/provider.py
=====================
BloombergDataProvider — couche haut niveau pour le projet equity-basket.

Portage de inspiration/equity-basket/src/backtest/bloomberg.py,
réécrit pour utiliser les nouvelles fonctions blp.bdh / blp.bdp / blp.bds.

Responsabilités :
    - Récupérer la composition historique de l'indice (point-in-time, anti-biais de survivant)
    - Télécharger prix, P/E, taux sans risque depuis Bloomberg
    - Sauvegarder en parquet → évite de re-fetcher à chaque run
    - Recharger depuis cache si disponible (fetch_or_load)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from . import blp as _blp
    BLP_AVAILABLE = True
except ImportError:
    BLP_AVAILABLE = False
    warnings.warn(
        "blpapi non disponible. Assurez-vous que blpapi est installé "
        "et que Bloomberg Terminal est ouvert."
    )

# Correspondance code court → ticker Bloomberg
INDEX_TICKERS: dict[str, str] = {
    "SPX": "SPX Index",
    "MID": "MID Index",
    "SML": "SML Index",
    "RTY": "RTY Index",
    "RIY": "RIY Index",
}

RISK_FREE_TICKER = "USGG3M Index"
DEFAULT_BENCHMARK_TICKER = "SPX Index"

DEFAULT_STATIC_FIELDS = [
    "ID_ISIN",
    "ID_BB_GLOBAL",
    "NAME",
    "CRNCY",
    "GICS_SECTOR_NAME",
]

EXPECTED_FILES = [
    "prices.parquet",
    "pe_ratios.parquet",
    "universe.parquet",
    "risk_free_US.parquet",
    "benchmark.parquet",
    "static_metadata.parquet",
]


class BloombergDataProvider:
    """
    Fournisseur de données Bloomberg pour un indice actions.

    Récupère (ou charge depuis cache parquet) :
        - Composition historique mensuelle point-in-time
        - Prix ajustés quotidiens (PX_LAST)
        - P/E trailing (PE_RATIO)
        - Taux sans risque (T-Bill 3M — USGG3M Index)

    Paramètres
    ----------
    index_name   : code court de l'indice — 'SPX', 'MID', 'SML', 'RTY', 'RIY'
    start_date   : date de début (str 'YYYY-MM-DD')
    end_date     : date de fin
    output_path  : dossier de cache parquet (None = pas de cache)

    Exemple
    -------
    provider = BloombergDataProvider(
        index_name="SPX",
        start_date="2000-01-01",
        end_date="2025-12-31",
        output_path="data/raw",
    )
    data = provider.fetch_or_load()
    prices    = data['prices']       # DataFrame dates × tickers
    pe_ratios = data['pe_ratios']    # DataFrame dates × tickers
    universe  = data['universe']     # DataFrame date | ticker
    risk_free = data['risk_free']    # Series dates
    """

    def __init__(
        self,
        index_name: str = "SPX",
        start_date: str = "2000-01-01",
        end_date: str = "2025-12-31",
        benchmark_ticker: str | None = None,
        output_path: str | Path | None = "data/raw",
    ) -> None:
        if index_name not in INDEX_TICKERS:
            raise ValueError(
                f"Indice inconnu: '{index_name}'. "
                f"Disponibles: {list(INDEX_TICKERS)}"
            )
        self.index_name = index_name
        self.start_date = start_date
        self.end_date = end_date
        self.output_path = Path(output_path) if output_path else None
        self.index_ticker = INDEX_TICKERS[index_name]
        self.benchmark_ticker = benchmark_ticker or self.index_ticker

    # ─────────────────────────── Composition de l'indice ─────────────────────

    def get_index_members(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Composition de l'indice à une date donnée (ou actuelle si date=None).

        Utilise le champ BDS 'INDX_MWEIGHT_HIST' avec override END_DATE_OVERRIDE
        pour récupérer la composition point-in-time.

        Retourne
        --------
        pd.DataFrame avec une colonne 'ticker'
        """
        if not BLP_AVAILABLE:
            raise ImportError("blpapi non disponible.")

        try:
            if date:
                dt_str = date.replace("-", "")
                raw = _blp.bds(
                    self.index_ticker,
                    "INDX_MWEIGHT_HIST",
                    END_DATE_OVERRIDE=dt_str,
                )
            else:
                raw = _blp.bds(self.index_ticker, "INDX_MEMBERS")

            if raw.empty:
                return pd.DataFrame(columns=["ticker"])

            # Détecte la colonne contenant les tickers Bloomberg
            cols_lower = {str(c).lower(): c for c in raw.columns}
            ticker_col = None
            for key in [
                "index_member",
                "member_ticker_and_exchange_code",
                "index member",
                "member",
                "ticker",
            ]:
                if key in cols_lower:
                    ticker_col = cols_lower[key]
                    break

            if ticker_col is None:
                raise ValueError(
                    f"Colonne ticker introuvable. Colonnes disponibles: {list(raw.columns)}"
                )

            tickers = (
                raw[ticker_col]
                .apply(lambda x: x[0] if isinstance(x, (tuple, list)) else x)
                .astype(str)
                .str.strip()
            )
            tickers = tickers[
                tickers.notna()
                & (tickers != "")
                & (tickers.str.lower() != "nan")
            ]
            return pd.DataFrame({"ticker": tickers.values})

        except Exception as exc:
            print(f"Erreur composition {self.index_ticker} @ {date}: {exc}")
            return pd.DataFrame(columns=["ticker"])

    def get_historical_composition(self, dates: List[str]) -> pd.DataFrame:
        """
        Composition mensuelle point-in-time pour une liste de dates.

        Boucle sur chaque date de rebalancement et appelle
        get_index_members() pour construire un panel point-in-time
        (pas de biais de survivant).

        Retourne
        --------
        pd.DataFrame avec colonnes 'date' et 'ticker'
        """
        frames: list[pd.DataFrame] = []
        for i, date in enumerate(dates):
            members = self.get_index_members(date)
            if not members.empty:
                members = members.copy()
                members["date"] = pd.to_datetime(date)
                members["index_membership_flag"] = 1
                frames.append(members[["date", "ticker", "index_membership_flag"]])
            if (i + 1) % 12 == 0:
                print(
                    f"  Composition: {i + 1}/{len(dates)} dates traitées..."
                )

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", "index_membership_flag"])
        return pd.concat(frames, ignore_index=True)

    # ─────────────────────────── Séries temporelles ───────────────────────────

    def get_prices(
        self,
        tickers: List[str],
        field: str = "PX_LAST",
        adjustments: bool = True,
    ) -> pd.DataFrame:
        """
        Données journalières pour une liste de tickers (prix, P/E, etc.).

        Utilise blp.bdh et aplatit le MultiIndex (ticker, field) → DataFrame
        dates × tickers, avec noms de colonnes nettoyés (strip ' Equity').

        Retourne
        --------
        pd.DataFrame  index=DatetimeIndex, colonnes=tickers (format court)
        """
        if not BLP_AVAILABLE:
            raise ImportError("blpapi non disponible.")

        raw = _blp.bdh(
            tickers=tickers,
            flds=[field],
            start_date=self.start_date,
            end_date=self.end_date,
            adjustments=adjustments,
        )

        if raw.empty:
            return pd.DataFrame()

        # Extrait le niveau field du MultiIndex → DataFrame dates × tickers
        df = raw.droplevel("field", axis=1)

        # Nettoie les noms de colonnes (enlève ' US Equity', ' Equity')
        df.columns = [
            c.replace(" US Equity", "").replace(" Equity", "")
            for c in df.columns
        ]
        return df

    def get_pe_ratios(self, tickers: List[str]) -> pd.DataFrame:
        """P/E trailing (PE_RATIO) pour les tickers."""
        return self.get_prices(tickers, field="PE_RATIO", adjustments=False)

    def get_benchmark(
        self,
        ticker: str = DEFAULT_BENCHMARK_TICKER,
        field: str = "PX_LAST",
    ) -> pd.Series:
        """
        Série benchmark quotidienne (par défaut SPX Index).

        Les prix sont récupérés en version ajustée (dividendes/splits).
        """
        if not BLP_AVAILABLE:
            raise ImportError("blpapi non disponible.")

        raw = _blp.bdh(
            tickers=[ticker],
            flds=[field],
            start_date=self.start_date,
            end_date=self.end_date,
            adjustments=True,
        )

        if raw.empty:
            return pd.Series(dtype=float, name="benchmark")

        series = raw.droplevel("field", axis=1).iloc[:, 0]
        series.name = "benchmark"
        return series

    def get_static_metadata(
        self,
        tickers: List[str],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Mapping statique instrument-level (ISIN, BBGID, nom, devise, secteur).

        Retourne un DataFrame avec :
        - ticker_bbg : ticker Bloomberg complet
        - ticker     : ticker court (sans suffixe Equity)
        - colonnes de metadata demandées
        """
        if not BLP_AVAILABLE:
            raise ImportError("blpapi non disponible.")

        fields = fields or DEFAULT_STATIC_FIELDS
        raw = _blp.bdp(tickers=tickers, flds=fields)

        cols = [f.lower() for f in fields]
        if raw.empty:
            return pd.DataFrame(columns=["ticker_bbg", "ticker", *cols])

        meta = raw.reset_index().rename(columns={"ticker": "ticker_bbg"})
        meta["ticker"] = (
            meta["ticker_bbg"]
            .astype(str)
            .str.replace(" US Equity", "", regex=False)
            .str.replace(" Equity", "", regex=False)
            .str.strip()
        )

        ordered_cols = ["ticker_bbg", "ticker", *cols]
        for c in ordered_cols:
            if c not in meta.columns:
                meta[c] = pd.NA

        return meta[ordered_cols]

    def get_risk_free_rate(self) -> pd.Series:
        """
        Taux sans risque journalier — T-Bill 3 mois (USGG3M Index).

        Retourne
        --------
        pd.Series  index=DatetimeIndex, name='risk_free'
        """
        if not BLP_AVAILABLE:
            raise ImportError("blpapi non disponible.")

        raw = _blp.bdh(
            tickers=[RISK_FREE_TICKER],
            flds=["PX_LAST"],
            start_date=self.start_date,
            end_date=self.end_date,
        )

        if raw.empty:
            return pd.Series(dtype=float, name="risk_free")

        series = raw.droplevel("field", axis=1).iloc[:, 0]
        series.name = "risk_free"
        return series

    # ─────────────────────────── Pipeline complet ─────────────────────────────

    def fetch_all(
        self,
        rebalance_dates: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch complet depuis Bloomberg :
        composition + prix + benchmark + mapping statique + P/E + taux sans risque.

        Paramètres
        ----------
        rebalance_dates : liste de dates 'YYYY-MM-DD' pour la composition
                          (None = composition à end_date uniquement)

        Retourne
        --------
        dict avec clés :
        - 'prices'
        - 'benchmark'
        - 'static_metadata'
        - 'pe_ratios'
        - 'universe'
        - 'risk_free'
        """
        if not BLP_AVAILABLE:
            raise ImportError("blpapi non disponible.")

        # Composition historique
        if rebalance_dates:
            universe = self.get_historical_composition(rebalance_dates)
        else:
            members = self.get_index_members()
            universe = pd.DataFrame({
                "date": pd.to_datetime(self.end_date),
                "ticker": members["ticker"],
                "index_membership_flag": 1,
            })

        if universe.empty:
            return {}

        raw_tickers = universe["ticker"].unique().tolist()

        # Ajoute ' Equity' si absent (convention Bloomberg)
        tickers_bbg = [
            t if t.endswith(" Equity") else f"{t} Equity"
            for t in raw_tickers
        ]

        print(f"Fetch Bloomberg : {len(tickers_bbg)} tickers | "
              f"{self.start_date} → {self.end_date}")

        prices    = self.get_prices(tickers_bbg, field="PX_LAST")
        benchmark = self.get_benchmark(ticker=self.benchmark_ticker, field="PX_LAST")
        static_metadata = self.get_static_metadata(tickers_bbg)
        pe_ratios = self.get_pe_ratios(tickers_bbg)
        risk_free = self.get_risk_free_rate()

        return {
            "prices":    prices,
            "benchmark": benchmark,
            "static_metadata": static_metadata,
            "pe_ratios": pe_ratios,
            "universe":  universe,
            "risk_free": risk_free,
        }

    def data_exists(self) -> bool:
        """Retourne True si tous les fichiers parquet du cache sont présents."""
        if self.output_path is None:
            return False
        return all(
            (self.output_path / f).exists() for f in EXPECTED_FILES
        )

    def save(self, data: Dict) -> None:
        """Sauvegarde les datasets en parquet dans output_path."""
        if self.output_path is None:
            raise ValueError("output_path non défini.")
        self.output_path.mkdir(parents=True, exist_ok=True)

        data["prices"].to_parquet(self.output_path / "prices.parquet")
        data["pe_ratios"].to_parquet(self.output_path / "pe_ratios.parquet")
        data["universe"].to_parquet(self.output_path / "universe.parquet")
        data["static_metadata"].to_parquet(self.output_path / "static_metadata.parquet")

        rf = data["risk_free"]
        if isinstance(rf, pd.Series):
            rf = rf.to_frame("risk_free")
        rf.to_parquet(self.output_path / "risk_free_US.parquet")

        bench = data["benchmark"]
        if isinstance(bench, pd.Series):
            bench = bench.to_frame("benchmark")
        bench.to_parquet(self.output_path / "benchmark.parquet")

        print(f"Cache sauvegardé dans {self.output_path}/")

    def load(self) -> Dict[str, pd.DataFrame]:
        """Charge les datasets depuis le cache parquet."""
        if self.output_path is None:
            raise ValueError("output_path non défini.")

        prices    = pd.read_parquet(self.output_path / "prices.parquet")
        pe_ratios = pd.read_parquet(self.output_path / "pe_ratios.parquet")
        universe  = pd.read_parquet(self.output_path / "universe.parquet")
        rf_df     = pd.read_parquet(self.output_path / "risk_free_US.parquet")

        static_path = self.output_path / "static_metadata.parquet"
        bench_path = self.output_path / "benchmark.parquet"

        static_metadata = (
            pd.read_parquet(static_path)
            if static_path.exists()
            else pd.DataFrame(columns=["ticker_bbg", "ticker", *[f.lower() for f in DEFAULT_STATIC_FIELDS]])
        )

        benchmark_df = (
            pd.read_parquet(bench_path)
            if bench_path.exists()
            else pd.DataFrame(columns=["benchmark"])
        )

        risk_free = rf_df.iloc[:, 0]
        risk_free.name = "risk_free"

        benchmark = (
            benchmark_df.iloc[:, 0]
            if len(benchmark_df.columns) > 0
            else pd.Series(dtype=float, name="benchmark")
        )
        benchmark.name = "benchmark"

        return {
            "prices":    prices,
            "benchmark": benchmark,
            "static_metadata": static_metadata,
            "pe_ratios": pe_ratios,
            "universe":  universe,
            "risk_free": risk_free,
        }

    def fetch_or_load(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Charge depuis le cache parquet si disponible, sinon fetch Bloomberg.

        Paramètres
        ----------
        force_refresh : si True, toujours re-fetch même si le cache existe

        Retourne
        --------
        dict avec clés :
        'prices', 'benchmark', 'static_metadata',
        'pe_ratios', 'universe', 'risk_free'
        """
        if not force_refresh and self.data_exists():
            print(
                f"Cache trouvé dans {self.output_path}/ "
                "— chargement depuis le disque."
            )
            return self.load()

        print(
            f"Données absentes dans {self.output_path}/ "
            "— fetch depuis Bloomberg..."
        )

        # Dates de rebalancement mensuelles sur toute la période
        rebalance_dates = (
            pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq="ME",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )

        data = self.fetch_all(rebalance_dates=rebalance_dates)

        if data and self.output_path:
            self.save(data)

        return data
