"""Standalone data check - no streamlit needed.

    python check_data.py            # checks the folder this file sits in
    python check_data.py /some/dir  # checks that folder

Prints what the dashboard will see: which CSVs are found, each index's date
range, and anything missing. Run this first whenever the dashboard looks stale.
"""
import os
import sys
import glob

import pandas as pd

TARGET = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__)
)

print(f"folder: {TARGET}\n")
files = sorted(glob.glob(os.path.join(TARGET, "*.csv")))
if not files:
    print("No .csv files here at all. The dashboard will show nothing.")
    sys.exit(1)

rows, loaded, rejected = [], [], []
for f in files:
    base = os.path.basename(f)
    try:
        d = pd.read_csv(f)
    except Exception as e:
        rejected.append((base, f"unreadable: {e}"))
        continue
    d.columns = d.columns.str.strip().str.replace('"', "", regex=False)
    if not {"P/E", "P/B"}.issubset(d.columns):
        rejected.append((base, "no P/E + P/B columns (price/PR export?)"))
        continue
    if "Date" not in d.columns:
        rejected.append((base, "no Date column"))
        continue
    name_col = "IndexName" if "IndexName" in d.columns else (
        "Index Name" if "Index Name" in d.columns else None
    )
    nm = (
        d[name_col].astype(str).str.strip().str.replace('"', "", regex=False)
        .str.replace(r"\s+", " ", regex=True).str.upper()
        if name_col else pd.Series([base.split("_Historical")[0].upper()] * len(d))
    )
    dt = pd.to_datetime(
        d["Date"].astype(str).str.strip().str.replace('"', "", regex=False),
        format="mixed", dayfirst=True, errors="coerce",
    )
    ok = dt.notna()
    if not ok.any():
        rejected.append((base, "no parseable dates"))
        continue
    loaded.append((base, len(d), dt[ok].min().date(), dt[ok].max().date()))
    rows.append(pd.DataFrame({"Index Name": nm[ok], "Date": dt[ok]}))

print(f"LOADED ({len(loaded)} files):")
for base, n, lo, hi in loaded:
    print(f"  {base}  ({n} rows, {lo} -> {hi})")

if rejected:
    print(f"\nNOT LOADED ({len(rejected)} files):")
    for base, why in rejected:
        print(f"  {base}  <- {why}")

if not rows:
    print("\nNothing usable found.")
    sys.exit(1)

all_rows = pd.concat(rows, ignore_index=True).drop_duplicates(
    subset=["Index Name", "Date"]
)
print(f"\nMERGED: {len(all_rows)} unique (index, date) rows")
print(f"overall range: {all_rows['Date'].min().date()} -> {all_rows['Date'].max().date()}\n")

per = all_rows.groupby("Index Name")["Date"].agg(["min", "max", "size"])
per.columns = ["From", "To", "Rows"]
overall_max = per["To"].max()
per["Lag (d)"] = (overall_max - per["To"]).dt.days
newest = all_rows["Date"].dt.to_period("M").max()
per[f"{newest.strftime('%b %Y')} days"] = (
    all_rows[all_rows["Date"].dt.to_period("M") == newest]
    .groupby("Index Name").size().reindex(per.index).fillna(0).astype(int)
)
print(per.to_string())

stale = per[per["Lag (d)"] > 0]
if not stale.empty:
    print(f"\n>>> {len(stale)} index(es) do NOT reach {overall_max.date()}:")
    for n, r in stale.iterrows():
        print(f"      {n}: stops {r['To'].date()} ({r['Lag (d)']}d behind)")
    print("    The export covering those dates is missing from this folder.")
else:
    print(f"\nEvery index reaches {overall_max.date()}. The data here is complete.")
    print("If the dashboard still shows an older date, it is not reading THIS")
    print("folder, or it is serving a cached result - fully restart streamlit.")
