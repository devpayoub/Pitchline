"""
Honest out-of-time evaluation of the models that are on disk.

    python scripts/backtest_models.py

For every league it sorts the processed matches by date, trains on the first
80% and scores the last 20% — data the model never saw, in a period after the
training window. That is the only way to get a number that is not inflated by
the leakage the pipeline used to have.

Output: data/history/backtest.json (read by the Accuracy tab).
"""
import os
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import features as features_mod      # noqa: E402

OUT = os.path.join(BASE, "data", "history", "backtest.json")

MARKETS = {
    "1x2": ("Result", 3),
    "ou15": ("Over1.5", 2),
    "ou25": ("Over2.5", 2),
    "ou35": ("Over3.5", 2),
    "btts": ("BTTS", 2),
}


def xgb():
    return XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.07,
        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=2,
    )


def evaluate_league(path, test_fraction=0.2):
    df = pd.read_csv(path, low_memory=False)
    if "Date" not in df.columns or len(df) < 200:
        return None
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    split = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:split], df.iloc[split:]
    feats = features_mod.feature_columns(df)
    if not feats or len(test) < 30:
        return None

    result = {"rows": int(len(df)), "test_rows": int(len(test)),
              "test_from": str(test["Date"].min().date()), "markets": {}}

    for market, (target, n_class) in MARKETS.items():
        if target not in df.columns:
            continue
        y_train = train[target].astype(int).values
        y_test = test[target].astype(int).values
        if len(np.unique(y_train)) < n_class:
            continue

        model = xgb()
        model.fit(train[feats].values, y_train)
        proba = model.predict_proba(test[feats].values)
        pred = proba.argmax(axis=1)

        acc = float((pred == y_test).mean())
        majority = float(pd.Series(y_test).value_counts(normalize=True).max())
        try:
            onehot = np.zeros_like(proba)
            onehot[np.arange(len(y_test)), y_test] = 1
            brier = float(((proba - onehot) ** 2).sum(axis=1).mean())
        except Exception:
            brier = None

        result["markets"][market] = {
            "accuracy": round(acc, 4),
            "baseline": round(majority, 4),   # always picking the most common class
            "lift": round(acc - majority, 4),
            "brier": round(brier, 4) if brier is not None else None,
            "n": int(len(y_test)),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=os.path.join(BASE, "data", "processed"))
    ap.add_argument("--test_fraction", type=float, default=0.2)
    args = ap.parse_args()

    per_league, pooled = {}, {}
    for path in sorted(glob.glob(os.path.join(args.processed_dir, "*_merged_preprocessed.csv"))):
        code = os.path.basename(path).split("_")[0]
        if code not in ("E0", "SP1", "I1", "D1", "F1", "CL"):
            continue
        print(f"Backtesting {code}...")
        res = evaluate_league(path, args.test_fraction)
        if not res:
            print("  skipped (not enough rows)")
            continue
        per_league[code] = res
        for market, m in res["markets"].items():
            acc = pooled.setdefault(market, {"correct": 0, "n": 0, "baseline": 0.0, "brier": 0.0, "leagues": 0})
            acc["correct"] += m["accuracy"] * m["n"]
            acc["n"] += m["n"]
            acc["baseline"] += m["baseline"] * m["n"]
            if m["brier"] is not None:
                acc["brier"] += m["brier"] * m["n"]
            acc["leagues"] += 1

    for market, a in pooled.items():
        n = max(a["n"], 1)
        a["accuracy"] = round(a["correct"] / n, 4)
        a["baseline"] = round(a["baseline"] / n, 4)
        a["lift"] = round(a["accuracy"] - a["baseline"], 4)
        a["brier"] = round(a["brier"] / n, 4)
        del a["correct"]

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method": f"train on the first {int((1 - args.test_fraction) * 100)}% of each league by date, "
                  f"score the last {int(args.test_fraction * 100)}%",
        "markets": pooled,
        "per_league": per_league,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nOut-of-time accuracy (pooled):")
    for market in ("1x2", "ou25", "ou15", "ou35", "btts"):
        if market in pooled:
            m = pooled[market]
            print(f"  {market:5s} {m['accuracy'] * 100:5.1f}%  vs baseline {m['baseline'] * 100:5.1f}%"
                  f"  (lift {m['lift'] * 100:+.1f} pts, n={m['n']})")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
