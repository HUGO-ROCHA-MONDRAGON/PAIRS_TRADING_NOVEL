"""
Export de portefeuille vers Bloomberg.
Génère des fichiers XLSX pour import dans Bloomberg PORT,
ainsi que les benchmarks et instructions de rebalancement en CSV.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

from utils import normalize_bbg_ticker


RAW_PATH    = Path("data/raw")
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def export_portfolio_xlsx(
    weights: pd.DataFrame,
    strategy_name: str        = "MomentumLongShort",
    track: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, float]] = None,
    currency: str             = "USD",
    output_path: Path         = OUTPUT_PATH,
) -> Path:
    """
    Exporte les poids du portefeuille vers un fichier Excel multi-feuilles pour Bloomberg PORT.
    Feuilles: Portfolio (poids), Metadata, Summary, Metrics, Top Long, Top Short.
    """
    # Feuille Portfolio - on vire les poids trop petits
    df = weights[["date", "ticker", "weight"]].copy()
    df = df[df["weight"].abs() > 1e-8]

    df["ticker"] = df["ticker"].apply(normalize_bbg_ticker)

    # Bloomberg PORT veut longs=100% et shorts=-100% séparément
    def _normalize_ls(grp):
        w = grp["weight"]
        long_sum  = w[w > 0].sum()
        short_sum = w[w < 0].sum()
        w_norm = w.copy()
        if long_sum > 0:
            w_norm[w > 0] = w[w > 0] / long_sum        # longs somment à 1.0
        if short_sum < 0:
            w_norm[w < 0] = w[w < 0] / abs(short_sum) * (-1)  # shorts somment à -1.0
        grp["weight"] = w_norm
        return grp

    df = df.groupby("date", group_keys=False).apply(_normalize_ls)

    df = df.rename(columns={
        "ticker": "Security ID",
        "weight": "Weight",
        "date":   "Date",
    })
    df["Portefeuille"] = strategy_name
    df["Date"]         = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Weight"]       = df["Weight"].round(6)
    df                 = df[["Security ID", "Weight", "Date", "Portefeuille"]]
    df                 = df.sort_values(["Date", "Security ID"]).reset_index(drop=True)

    # Feuille Metadata
    first_date = df["Date"].min()
    last_date  = df["Date"].max()
    metadata   = pd.DataFrame({
        "Property": [
            "Portfolio Name", "Currency", "Creation Date",
            "Rebalance Frequency", "Start Date", "End Date",
            "Total Observations", "Unique Securities",
        ],
        "Value": [
            strategy_name, currency, datetime.now().strftime("%Y-%m-%d"),
            "Monthly", first_date, last_date,
            len(df), df["Security ID"].nunique(),
        ],
    })

    # Feuille Summary (optionnelle)
    summary = None
    if track is not None:
        raw_weights = weights.copy()
        summary = raw_weights.groupby("date").apply(lambda x: pd.Series({
            "n_positions":    len(x),
            "n_long":         (x["weight"] > 0).sum(),
            "n_short":        (x["weight"] < 0).sum(),
            "long_exposure":  x[x["weight"] > 0]["weight"].sum(),
            "short_exposure": x[x["weight"] < 0]["weight"].sum(),
            "gross_exposure": x["weight"].abs().sum(),
            "net_exposure":   x["weight"].sum(),
        })).reset_index()

        if "date" in track.columns:
            cols = [c for c in ["date", "net_value", "net_return", "turnover"] if c in track.columns]
            summary = summary.merge(track[cols], on="date", how="left")

        if "net_value" in summary.columns:
            summary["cumulative_return_%"] = (
                summary["net_value"] / summary["net_value"].iloc[0] - 1
            ) * 100

        summary["date"] = pd.to_datetime(summary["date"]).dt.strftime("%Y-%m-%d")

    # Feuille Metrics (optionnelle)
    metrics_df = None
    if metrics:
        metrics_df = pd.DataFrame([
            {"Metric": "Total Return",       "Value": f"{metrics.get('total_return', 0)*100:.2f}%"},
            {"Metric": "Annualized Return",  "Value": f"{metrics.get('annualized_return', 0)*100:.2f}%"},
            {"Metric": "Volatility",         "Value": f"{metrics.get('volatility_annual', 0)*100:.2f}%"},
            {"Metric": "Sharpe Ratio",       "Value": f"{metrics.get('sharpe_ratio', 0):.3f}"},
            {"Metric": "Sortino Ratio",      "Value": f"{metrics.get('sortino_ratio', 0):.3f}"},
            {"Metric": "Max Drawdown",       "Value": f"{metrics.get('max_drawdown', 0)*100:.2f}%"},
            {"Metric": "VaR (95%)",          "Value": f"{metrics.get('var_95', 0)*100:.2f}%"},
            {"Metric": "CVaR (95%)",         "Value": f"{metrics.get('cvar_95', 0)*100:.2f}%"},
        ])

    # Feuilles Top positions
    raw_weights  = weights.copy()
    raw_weights  = raw_weights[raw_weights["weight"].abs() > 1e-8]
    last_rb_date = pd.to_datetime(raw_weights["date"]).max()
    last_w       = raw_weights[raw_weights["date"] == last_rb_date].copy()
    last_w["ticker"] = last_w["ticker"].apply(normalize_bbg_ticker)

    top_long  = (last_w.nlargest(10, "weight")[["ticker", "weight"]].copy()
                 .rename(columns={"ticker": "Security ID", "weight": "Weight (%)"}))
    top_long["Weight (%)"] *= 100

    top_short = (last_w.nsmallest(10, "weight")[["ticker", "weight"]].copy()
                 .rename(columns={"ticker": "Security ID", "weight": "Weight (%)"}))
    top_short["Weight (%)"] *= 100

    # Écriture du fichier Excel
    out_file = output_path / f"portfolio_{strategy_name}.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df.to_excel(writer,       sheet_name="Portfolio",  index=False)
        metadata.to_excel(writer, sheet_name="Metadata",   index=False)
        if summary is not None:
            summary.to_excel(writer,    sheet_name="Summary",    index=False)
        if metrics_df is not None:
            metrics_df.to_excel(writer, sheet_name="Metrics",    index=False)
        top_long.to_excel(writer,  sheet_name="Top Long",   index=False)
        top_short.to_excel(writer, sheet_name="Top Short",  index=False)

    return out_file




def export_benchmark_csv(
    raw_path: Path    = RAW_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Convertit data/raw/benchmark.parquet en CSV."""
    parquet_file = raw_path / "benchmark.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"benchmark.parquet introuvable dans {raw_path}")

    bm = pd.read_parquet(parquet_file)
    if isinstance(bm, pd.Series):
        bm = bm.to_frame(name="benchmark")

    bm.index      = pd.to_datetime(bm.index)
    bm.index.name = "Date"
    bm.columns    = ["benchmark"]

    out_file = output_path / "benchmark.csv"
    bm.to_csv(out_file)

    return out_file


if __name__ == "__main__":
    # Dummy data pour tester
    dates   = pd.date_range("2024-01-31", "2024-12-31", freq="ME")
    tickers = ["AAPL UW", "MSFT UW", "NVDA UW", "GOOGL UW", "AMZN UW"]

    weights_list = [
        {"date": d, "ticker": t, "weight": round(1 / len(tickers), 6)}
        for d in dates for t in tickers
    ]
    weights = pd.DataFrame(weights_list)

    track = pd.DataFrame({
        "date":       dates,
        "net_value":  1_000_000 * (1 + np.random.normal(0.001, 0.01, len(dates))).cumprod(),
        "net_return": np.random.normal(0.001, 0.01, len(dates)),
        "turnover":   np.full(len(dates), 0.2),
    })

    metrics = {
        "total_return": 0.142, "annualized_return": 0.135,
        "volatility_annual": 0.18, "sharpe_ratio": 0.75,
        "sortino_ratio": 1.1, "max_drawdown": -0.12,
        "var_95": -0.02, "cvar_95": -0.03,
    }

    # Export
    export_portfolio_xlsx(weights, strategy_name="MomentumLongShort",
                          track=track, metrics=metrics)
    export_benchmark_csv()