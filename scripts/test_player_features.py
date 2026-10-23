"""
Does player availability actually help? Train on 2025-26, score 2026-27, with
and without the player features.

    python scripts/test_player_features.py

Same rows, same split, same model — the only difference is the feature set, so
the delta is attributable to the player data rather than to noise in the split.
"""
import os
import re
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import features as features_mod        # noqa: E402
from teams import side_similarity      # noqa: E402

PROCESSED = os.path.join(BASE, "data", "processed", "E0_merged_preprocessed.csv")
PLAYER_CSV = os.path.join(BASE, "data", "players", "e0_player_features.csv")
OUT = os.path.join(BASE, "data", "players", "feature_test.json")

PLAYER_COLS = ["xi_share", "rotation", "xi_goal_share", "missing_count", "missing_key"]
MARKETS = {"ou25": ("Over2.5", 2), "1x2": ("Result", 3), "btts": ("BTTS", 2)}


def parse_date(value):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(str(value)[:10], fmt)
        except ValueError:
            continue
    return None


def strip_country(name):
    """'Hull (Eng)' -> 'Hull'. The player cache spans every competition, and
    Flashscore tags clubs with a country once a page mixes leagues."""
    return re.sub(r"\s*\([A-Za-z]{2,3}\)\s*$", "", str(name)).strip()


def build_name_map(fs_names, data_names):
    """Flashscore club name -> the name football-data uses."""
    mapping = {}
    for fs in (strip_country(n) for n in fs_names):
        if fs in mapping:
            continue
        best, score = None, 0.0
        for name in data_names:
            s = side_similarity(name, fs)
            if s > score:
                best, score = name, s
        if score >= 0.8:
            mapping[fs] = best
    return mapping


def load_merged():
    df = pd.read_csv(PROCESSED, low_memory=False)
    df["_date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["_date"])

    pf = pd.read_csv(PLAYER_CSV)
    pf["_date"] = pf["date"].apply(parse_date)
    pf = pf.dropna(subset=["_date"])

    name_map = build_name_map(set(pf["home"]) | set(pf["away"]),
                              set(df["HomeTeam"]) | set(df["AwayTeam"]))
    pf["home_data"] = pf["home"].apply(strip_country).map(name_map)
    pf["away_data"] = pf["away"].apply(strip_country).map(name_map)

    keep = [c for c in pf.columns if any(c.startswith(p) for p in PLAYER_COLS)]
    pf_slim = pf[["_date", "home_data", "away_data"] + keep].dropna(subset=["home_data", "away_data"])
    pf_slim = pf_slim.drop_duplicates(subset=["_date", "home_data", "away_data"])

    merged = df.merge(pf_slim, left_on=["_date", "HomeTeam", "AwayTeam"],
                      right_on=["_date", "home_data", "away_data"], how="inner")
    print(f"processed rows: {len(df)} · player rows: {len(pf)} · matched: {len(merged)}")
    return merged, name_map


def run(merged):
    base_features = [c for c in features_mod.feature_columns(merged)
                     if not any(c.startswith(p) for p in PLAYER_COLS)]
    player_features = [c for c in merged.columns if any(c.startswith(p) for p in PLAYER_COLS)]

    train = merged[merged["_date"] < datetime(2026, 7, 1)]
    test = merged[merged["_date"] >= datetime(2026, 7, 1)]
    print(f"train (2025-26): {len(train)} · test (2026-27): {len(test)}")
    if len(train) < 100 or len(test) < 30:
        print("Not enough rows for a fair comparison.")
        return None

    results = {}
    for market, (target, n_class) in MARKETS.items():
        if target not in merged.columns:
            continue
        y_train = train[target].astype(int).values
        y_test = test[target].astype(int).values
        if len(np.unique(y_train)) < n_class:
            continue

        row = {}
        for label, cols in (("base", base_features),
                            ("base+player", base_features + player_features)):
            X_train = train[cols].astype(float).fillna(0).values
            X_test = test[cols].astype(float).fillna(0).values
            model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                                  subsample=0.85, colsample_bytree=0.85,
                                  eval_metric="logloss", tree_method="hist",
                                  random_state=42, n_jobs=2)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            acc = float((proba.argmax(axis=1) == y_test).mean())

            onehot = np.zeros_like(proba)
            onehot[np.arange(len(y_test)), y_test] = 1
            brier = float(((proba - onehot) ** 2).sum(axis=1).mean())

            row[label] = {"accuracy": round(acc, 4), "brier": round(brier, 4),
                          "features": len(cols)}
        row["delta_accuracy"] = round(row["base+player"]["accuracy"] - row["base"]["accuracy"], 4)
        row["delta_brier"] = round(row["base+player"]["brier"] - row["base"]["brier"], 4)
        results[market] = row

    payload = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "player_features": player_features, "markets": results,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'market':6s} {'base':>8s} {'+player':>9s} {'delta':>8s}   {'brier base':>11s} {'brier new':>10s}")
    for market, r in results.items():
        print(f"{market:6s} {r['base']['accuracy']*100:7.1f}% {r['base+player']['accuracy']*100:8.1f}% "
              f"{r['delta_accuracy']*100:+7.1f}pt   {r['base']['brier']:11.4f} {r['base+player']['brier']:10.4f}")
    print(f"\nSaved {OUT}")
    return payload


if __name__ == "__main__":
    merged, name_map = load_merged()
    run(merged)
