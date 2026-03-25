#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for p in (text or "").split(","):
        s = p.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("No seeds provided")
    return out


def _run_one(
    project_root: Path,
    train_script: Path,
    db: str,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch: int,
    val_split: float,
    lr: float,
    patience: int,
    min_delta: float,
) -> Path:
    cmd = [
        sys.executable,
        str(train_script),
        "--db",
        db,
        "--out",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--val_split",
        str(val_split),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--early_stopping",
        "--patience",
        str(patience),
        "--min_delta",
        str(min_delta),
    ]

    print(f"\n[multi_seed] Running seed={seed} -> {out_dir}")
    subprocess.run(cmd, check=True, cwd=str(project_root))

    report_path = out_dir / "training_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing training report: {report_path}")
    return report_path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _score(report: dict[str, Any]) -> tuple[float, float, float]:
    fm = dict(report.get("final_metrics") or {})
    val_auc = _safe_float(fm.get("val_auc"), -1.0)
    val_loss = _safe_float(fm.get("val_loss"), 1e9)
    val_acc = _safe_float(fm.get("val_acc"), -1.0)
    return (val_auc, -val_loss, val_acc)


def _publish_best(best_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.keras", "feature_mapping.json", "training_report.json"):
        src = best_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-seed TF training and pick best model")
    ap.add_argument("--db", default=str(Path("data") / "app.db"))
    ap.add_argument("--base_out", default=str(Path("models") / "tf_v3"))
    ap.add_argument("--seeds", default="11,42,77", help="Comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--min_delta", type=float, default=0.0003)
    ap.add_argument("--no_publish", action="store_true", help="Do not copy best artifacts to --base_out")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    train_script = project_root / "train_model.py"
    if not train_script.exists():
        raise SystemExit(f"Missing script: {train_script}")

    seeds = _parse_seeds(args.seeds)
    base_out = (project_root / Path(args.base_out)).resolve()

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        run_out = Path(f"{str(base_out)}_s{seed}")
        report_path = _run_one(
            project_root=project_root,
            train_script=train_script,
            db=args.db,
            out_dir=run_out,
            seed=seed,
            epochs=args.epochs,
            batch=args.batch,
            val_split=args.val_split,
            lr=args.lr,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        rep = _load_json(report_path)
        metrics = dict(rep.get("final_metrics") or {})
        runs.append(
            {
                "seed": seed,
                "out_dir": str(run_out),
                "report_path": str(report_path),
                "val_auc": _safe_float(metrics.get("val_auc"), -1.0),
                "val_loss": _safe_float(metrics.get("val_loss"), 1e9),
                "val_acc": _safe_float(metrics.get("val_acc"), -1.0),
                "score": _score(rep),
            }
        )

    if not runs:
        raise SystemExit("No runs completed")

    best = max(runs, key=lambda r: r["score"])
    best_dir = Path(str(best["out_dir"]))

    selection_report = {
        "base_out": str(base_out),
        "seeds": seeds,
        "selection_rule": "max val_auc, then min val_loss, then max val_acc",
        "runs": [
            {
                "seed": int(r["seed"]),
                "out_dir": r["out_dir"],
                "report_path": r["report_path"],
                "val_auc": float(r["val_auc"]),
                "val_loss": float(r["val_loss"]),
                "val_acc": float(r["val_acc"]),
            }
            for r in runs
        ],
        "best": {
            "seed": int(best["seed"]),
            "out_dir": best["out_dir"],
            "report_path": best["report_path"],
            "val_auc": float(best["val_auc"]),
            "val_loss": float(best["val_loss"]),
            "val_acc": float(best["val_acc"]),
        },
    }

    selection_path = base_out.parent / f"{base_out.name}_seed_selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection_report, f, ensure_ascii=False, indent=2)

    if not args.no_publish:
        _publish_best(best_dir, base_out)
        print(f"\n[multi_seed] Published best run (seed={best['seed']}) to: {base_out}")

    print("\n[multi_seed] Completed")
    print(f"[multi_seed] Best seed: {best['seed']}")
    print(f"[multi_seed] Best val_auc: {best['val_auc']:.6f}")
    print(f"[multi_seed] Best val_loss: {best['val_loss']:.6f}")
    print(f"[multi_seed] Best val_acc: {best['val_acc']:.6f}")
    print(f"[multi_seed] Selection report: {selection_path}")


if __name__ == "__main__":
    main()
