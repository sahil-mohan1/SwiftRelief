#!/usr/bin/env python3
from pathlib import Path
import sys
p = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(p))
from train_model_old import fetch_training_rows, build_vocab, vectorize
import sqlite3
con = sqlite3.connect('file:data/app.db?mode=ro', uri=True)
rows = fetch_training_rows(con)
con.close()
if not rows:
    print('NO_ROWS')
else:
    vocab = build_vocab(rows)
    X, y = vectorize(rows, vocab)
    print('ROWS', len(rows), 'DIM', X.shape[1])
