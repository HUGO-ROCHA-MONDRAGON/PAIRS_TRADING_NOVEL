from pathlib import Path
import pandas as pd

base = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel\src\data\raw")
files = [
    "prices.parquet",
    "pe_ratios.parquet",
    "risk_free.parquet",
    "universe.parquet",
]

print(f"base={base}")
print("=" * 100)

for name in files:
    p = base / name
    print(f"\nFILE: {name}")
    if not p.exists():
        print("  -> MISSING")
        continue
    size_mb = p.stat().st_size / (1024 * 1024)
    print(f"  size_mb={size_mb:.2f}")

    df = pd.read_parquet(p)
    print(f"  shape={df.shape}")

    idx = df.index
    print(f"  index_type={type(idx).__name__}, index_name={idx.name}")
    if len(idx) > 0:
        try:
            idx_dt = pd.to_datetime(idx, errors="coerce")
            if hasattr(idx_dt, "notna") and idx_dt.notna().any():
                print(f"  index_min={idx_dt.min()} | index_max={idx_dt.max()}")
        except Exception:
            pass

    cols = list(df.columns)
    print(f"  n_columns={len(cols)}")
    if len(cols) > 0:
        print(f"  first_columns={cols[:10]}")

    dtypes_counts = df.dtypes.astype(str).value_counts()
    print("  dtypes_count=")
    for t, c in dtypes_counts.items():
        print(f"    - {t}: {c}")

    na_total = int(df.isna().sum().sum())
    total_vals = int(df.shape[0] * max(df.shape[1], 1))
    na_pct = (100.0 * na_total / total_vals) if total_vals > 0 else 0.0
    print(f"  missing_values={na_total} ({na_pct:.2f}%)")

    print("  preview=")
    if name in {"prices.parquet", "pe_ratios.parquet"}:
        if len(df) > 0 and len(df.columns) > 0:
            sub = df.iloc[:3, :6]
            print(sub.to_string())
        else:
            print("    <empty>")
    else:
        print(df.head(8).to_string(index=True))

print("\nDONE")
