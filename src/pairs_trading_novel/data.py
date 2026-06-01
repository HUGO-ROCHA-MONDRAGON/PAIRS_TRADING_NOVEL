"""
data.py — DataLoader OOP pour le pipeline Pairs Trading.

Charge les parquets bruts attendus par les modules de signal et de backtest.
Fournit un fallback yfinance optionnel si les fichiers Bloomberg ne sont pas
disponibles localement (utile en dev container sans terminal Bloomberg).

Fichiers attendus (sous src/data/raw/):
    - log_prices.parquet          (index dates × colonnes tickers, log(PX_LAST))
    - daily_returns.parquet       (rendements arithmétiques simples)
    - universe.parquet            (colonnes 'date', 'ticker' — composition pt-in-time)
    - benchmark.parquet           (1 colonne : niveau S&P 500)
    - risk_free_US.parquet        (1 colonne : USGG3M en % annualisé)
    - rebalance_dates.parquet     (1 colonne : 77 fins de mois 2017‑01 → 2023‑05)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import raw_data_dir


REQUIRED_FILES = [
    "log_prices.parquet",
    "daily_returns.parquet",
    "universe.parquet",
    "benchmark.parquet",
    "risk_free_US.parquet",
    "rebalance_dates.parquet",
]


@dataclass
class MarketData:
    """Conteneur immuable des séries temporelles utilisées par le backtest."""

    log_prices: pd.DataFrame
    daily_returns: pd.DataFrame
    universe: pd.DataFrame
    benchmark: pd.Series
    risk_free: pd.Series
    rebalance_dates: pd.DatetimeIndex
    meta: dict = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return list(self.log_prices.columns)

    @property
    def n_rebalances(self) -> int:
        return int(len(self.rebalance_dates))

    def coverage(self, date: pd.Timestamp, lookback_days: int = 504) -> pd.Series:
        hist = self.log_prices[self.log_prices.index <= date].tail(lookback_days)
        return hist.notna().mean()


class DataLoader:
    """Charge les parquets requis. Si manquants, peut tenter yfinance fallback."""

    def __init__(self, raw_dir: str | Path | None = None) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir else raw_data_dir()

    # ------------------------------------------------------------------ status
    def missing_files(self) -> list[str]:
        missing = []
        for name in REQUIRED_FILES:
            if name == "risk_free_US.parquet":
                if not (self.raw_dir / "risk_free_US.parquet").exists() and not (self.raw_dir / "Risk_free_US.xlsx").exists():
                    missing.append(name)
                continue
            if not (self.raw_dir / name).exists():
                missing.append(name)
        return missing

    def is_ready(self) -> bool:
        return not self.missing_files()

    # -------------------------------------------------------------------- load
    def load(self) -> MarketData:
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(
                f"Fichiers raw manquants dans {self.raw_dir}: {missing}. "
                "Utilisez DataLoader.bootstrap_from_yfinance() pour générer un "
                "jeu de données de secours, ou lancez l'ingestion Bloomberg "
                "(notebook 01)."
            )
        rp = self.raw_dir
        log_prices = pd.read_parquet(rp / "log_prices.parquet")
        daily_returns = pd.read_parquet(rp / "daily_returns.parquet")
        universe = pd.read_parquet(rp / "universe.parquet")
        benchmark_raw = pd.read_parquet(rp / "benchmark.parquet")
        risk_free_raw = self._load_risk_free_us(rp)
        rebalance_dates = pd.read_parquet(rp / "rebalance_dates.parquet")
        for df in (log_prices, daily_returns, benchmark_raw, risk_free_raw):
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
        universe["date"] = pd.to_datetime(universe["date"])
        benchmark = benchmark_raw.iloc[:, 0].pct_change().rename("benchmark")
        risk_free = ((risk_free_raw.iloc[:, 0] / 100.0) / 252.0).rename("rf_daily")
        return MarketData(
            log_prices=log_prices,
            daily_returns=daily_returns,
            universe=universe,
            benchmark=benchmark,
            risk_free=risk_free,
            rebalance_dates=pd.DatetimeIndex(pd.to_datetime(rebalance_dates.iloc[:, 0].values)),
            meta={"source": "parquet", "raw_dir": str(self.raw_dir)},
        )

    def _load_risk_free_us(self, raw_path: Path) -> pd.DataFrame:
        rf_parquet = raw_path / "risk_free_US.parquet"
        rf_xlsx = raw_path / "Risk_free_US.xlsx"
        if rf_parquet.exists():
            return pd.read_parquet(rf_parquet)
        if rf_xlsx.exists():
            df = pd.read_excel(rf_xlsx)
            if df.empty:
                raise ValueError("Risk_free_US.xlsx est vide.")
            # Heuristique simple: première colonne datetime, deuxième colonne valeur.
            if df.shape[1] < 2:
                raise ValueError("Risk_free_US.xlsx doit contenir au moins 2 colonnes (date, valeur).")
            date_col = df.columns[0]
            value_col = df.columns[1]
            out = df[[date_col, value_col]].copy()
            out.columns = ["date", "risk_free_us"]
            out["date"] = pd.to_datetime(out["date"])
            out = out.set_index("date").sort_index()
            out.to_parquet(rf_parquet)
            return out
        raise FileNotFoundError(
            f"Aucune source risk-free_US trouvée dans {raw_path}: attendu risk_free_US.parquet ou Risk_free_US.xlsx"
        )

    # --------------------------------------------------------------- fallback
    def bootstrap_from_yfinance(
        self,
        tickers: list[str] | None = None,
        start: str = "2014-12-01",
        end: str = "2023-06-15",
        benchmark: str = "^GSPC",
        risk_free_ticker: str = "^IRX",
    ) -> MarketData:
        """
        Fallback: télécharge un univers fixe depuis Yahoo Finance et reproduit
        la structure attendue. NB : ne respecte pas la composition point‑in‑time
        de l'indice (utilise un snapshot actuel) — biais de survivant.
        """
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Installez yfinance: pip install yfinance") from exc

        if tickers is None:
            tickers = _DEFAULT_SP100_FALLBACK

        self.raw_dir.mkdir(parents=True, exist_ok=True)

        raw = yf.download(
            tickers + [benchmark, risk_free_ticker],
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        # yfinance can return either a "Close"/"Adj Close" level — auto_adjust=True
        # collapses to OHLCV; we want only the Close prices
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = tickers + [benchmark, risk_free_ticker]
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().ffill(limit=2)

        bench_levels = prices[benchmark].rename("level")
        rf_series = prices[risk_free_ticker].rename("rf_annual_pct")

        asset_prices = prices.drop(columns=[benchmark, risk_free_ticker]).dropna(axis=1, how="all")
        log_prices = np.log(asset_prices.where(asset_prices > 0))
        daily_returns = asset_prices.pct_change()

        # Universe = snapshot répété à chaque fin de mois
        reb_dates = pd.date_range("2017-01-31", "2023-05-31", freq="ME")
        # Aligner sur les jours de bourse réellement disponibles
        reb_dates = pd.DatetimeIndex([prices.index[prices.index <= d][-1] for d in reb_dates if (prices.index <= d).any()])
        universe_rows = []
        for d in reb_dates:
            for t in asset_prices.columns:
                universe_rows.append({"date": d, "ticker": t})
        universe = pd.DataFrame(universe_rows)

        # Sauvegarde parquet pour rejouabilité
        log_prices.to_parquet(self.raw_dir / "log_prices.parquet")
        daily_returns.to_parquet(self.raw_dir / "daily_returns.parquet")
        universe.to_parquet(self.raw_dir / "universe.parquet")
        bench_levels.to_frame().to_parquet(self.raw_dir / "benchmark.parquet")
        rf_series.to_frame().to_parquet(self.raw_dir / "risk_free_US.parquet")
        pd.DataFrame({"date": reb_dates}).to_parquet(self.raw_dir / "rebalance_dates.parquet")

        benchmark_ret = bench_levels.pct_change().rename("benchmark")
        rf_daily = ((rf_series / 100.0) / 252.0).rename("rf_daily")

        return MarketData(
            log_prices=log_prices,
            daily_returns=daily_returns,
            universe=universe,
            benchmark=benchmark_ret,
            risk_free=rf_daily,
            rebalance_dates=reb_dates,
            meta={"source": "yfinance", "tickers": len(asset_prices.columns)},
        )


# Univers fallback de secours (S&P 100 ~ pour avoir suffisamment de candidats)
_DEFAULT_SP100_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "TSLA", "UNH",
    "XOM", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "LLY", "AVGO", "COST", "WMT", "ADBE", "MCD", "CSCO", "BAC",
    "ACN", "CRM", "TMO", "ABT", "PFE", "NFLX", "LIN", "DHR", "DIS", "WFC",
    "VZ", "TXN", "NEE", "AMD", "PM", "BMY", "RTX", "COP", "QCOM", "SCHW",
    "UNP", "HON", "LOW", "INTC", "T", "ORCL", "UPS", "INTU", "IBM", "AMGN",
    "GS", "CAT", "SPGI", "MDT", "BA", "BLK", "DE", "AXP", "GE", "ELV",
    "MS", "ISRG", "LMT", "SBUX", "PLD", "SYK", "MMC", "ADP", "TJX", "GILD",
    "VRTX", "CI", "AMT", "TMUS", "C", "MO", "BKNG", "ZTS", "REGN", "MDLZ",
    "SO", "DUK", "PYPL", "BDX", "CB", "EOG", "CL", "EQIX", "SLB", "AON",
    "ICE", "APD", "CME", "ITW", "NSC", "FDX", "EMR", "PNC", "USB", "TGT",
    "ETN", "MMM", "FCX", "MU", "GM", "F", "COF", "MET", "AIG", "PRU",
    "WBA", "KMB", "HUM", "ADI", "LRCX", "KLAC", "MAR", "MNST", "ROP", "PGR",
    "CTAS", "PSA", "AEP", "EXC", "SRE", "D", "PCAR", "ROST", "MCO", "AFL",
    "STZ", "TRV", "WM", "DG", "DLR", "PSX", "VLO", "MPC", "OXY", "HES",
]
