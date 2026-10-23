"""
Train lightweight XGBoost models for additional prediction targets:
  BTTS (Both Teams To Score), Over1.5, Over3.5

Usage:
    python scripts/train_additional_targets.py --processed_dir data/processed --models_dir models
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as features_mod

# Targets to train
ADDITIONAL_TARGETS = ["BTTS", "Over1.5", "Over3.5"]

# The leagues that have prediction models
LEAGUES = ["E0", "SP1", "I1", "D1", "F1", "CL"]

# EL reuses CL model
MODEL_ALIAS = {"EL": "CL"}


def load_data(processed_dir: str, league: str) -> pd.DataFrame:
    """Load processed CSV for a league."""
    model_league = MODEL_ALIAS.get(league, league)
    path = os.path.join(processed_dir, f"{model_league}_merged_preprocessed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target: str):
    """Return X, y for training, excluding every target column."""
    feature_cols = features_mod.feature_columns(df)
    X = df[feature_cols].values
    y = df[target].values
    return X, y, feature_cols


def train_and_save(X, y, feature_cols, league: str, target: str, models_dir: str):
    """Train a single XGBoost model and save it."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=2,
    )
    model.fit(X, y)

    # Quick CV check
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=2)
    print(f"  {league} {target}: accuracy = {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Save
    suffix = f"{target.lower().replace('.', '')}_classifier"
    filename = os.path.join(models_dir, f"{league}_{suffix}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    features_mod.save_manifest(models_dir, league, suffix, feature_cols,
                               extra={"target": target})
    print(f"    -> saved {filename} ({len(feature_cols)} features)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='data/processed')
    parser.add_argument('--models_dir', type=str, default='models')
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    for league in LEAGUES:
        print(f"\n{'='*50}")
        print(f"League: {league}")
        print('='*50)

        try:
            df = load_data(args.processed_dir, league)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        for target in ADDITIONAL_TARGETS:
            if target not in df.columns:
                print(f"  Target '{target}' not found, skipping")
                continue

            print(f"\n  Training {target}...")
            X, y, features = prepare_features(df, target)
            if len(np.unique(y)) < 2:
                print(f"    Only one class present, skipping")
                continue

            train_and_save(X, y, features, league, target, args.models_dir)

    print("\n✅ All additional models trained!")


if __name__ == "__main__":
    main()
