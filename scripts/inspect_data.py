from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading_novel.paths import raw_data_dir


def describe_parquet(path: Path) -> dict:
    df = pd.read_parquet(path)
    result = {"file": path.name, "rows": df.shape[0], "cols": df.shape[1]}
    if isinstance(df.index, pd.DatetimeIndex):
        result["start"] = str(df.index.min().date())
        result["end"] = str(df.index.max().date())
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        result["start"] = str(dates.min().date())
        result["end"] = str(dates.max().date())
    else:
        result["start"] = ""
        result["end"] = ""
    result["missing_ratio"] = float(df.isna().mean().mean()) if df.size else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir(ROOT))
    args = parser.parse_args()
    files = sorted(args.raw_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in {args.raw_dir}")
    summary = pd.DataFrame([describe_parquet(path) for path in files])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
