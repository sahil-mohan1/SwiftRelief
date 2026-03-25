#!/usr/bin/env python3
"""Print class balance for combined CSV, DB rows, and inferred synthetic portion."""
from collections import Counter
from pathlib import Path
import sys
import sqlite3
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

# combined CSV
csv_path = root / "data" / "combined_real_synth.csv"
df = pd.read_csv(csv_path)
comb = Counter(df["label"].astype(int))
print(f"combined: {dict(comb)} n={len(df)}")

# DB rows using project helper
try:
    import train_model as tm
except Exception as e:
    print("ERROR importing train_model:", e)
    raise

con = sqlite3.connect(str(root / "data" / "app.db"))
rows = tm.fetch_training_rows(con, max_samples=0, sample_order="none")
con.close()
labels = [int((r.get("label") or 0)) for r in rows]
db = Counter(labels)
print(f"db: {dict(db)} n={len(labels)}")

# inferred synthetic
synth_n = len(df) - len(labels)
synth = {k: comb.get(k, 0) - db.get(k, 0) for k in set(comb) | set(db)}
print(f"synthetic (inferred): {dict(synth)} n={synth_n}")
