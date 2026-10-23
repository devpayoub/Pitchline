"""
One definition of "the feature matrix", shared by preprocessing, training and
prediction.

Training and prediction used to disagree: `train_models.py` dropped only
`Over2.5` and `Result`, leaving `BTTS`, `Over1.5` and `Over3.5` in the feature
matrix. Those three are derived from the same scoreline as the target, so the
model was reading the answer, and predictions then failed with
"X has 16 features, but LogisticRegression is expecting 19".

Every model is now saved with a `<league>_<kind>_features.json` manifest, so a
prediction run selects exactly the columns the model was trained on and fails
loudly (instead of silently) when the processed data no longer matches.
"""
import os
import json

# Columns that are outcomes of the match itself and must never be features.
ALL_TARGETS = ("Over2.5", "Result", "BTTS", "Over1.5", "Over3.5")


def feature_columns(df):
    """Numeric columns minus every target column, in dataframe order."""
    numeric = df.select_dtypes(include=["number"]).columns
    return [c for c in numeric if c not in ALL_TARGETS]


def manifest_path(models_dir, league, kind):
    return os.path.join(models_dir, f"{league}_{kind}_features.json")


def save_manifest(models_dir, league, kind, features, extra=None):
    payload = {"league": league, "kind": kind, "features": list(features)}
    if extra:
        payload.update(extra)
    path = manifest_path(models_dir, league, kind)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_manifest(models_dir, league, kind):
    path = manifest_path(models_dir, league, kind)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("features")


def resolve_feature_list(df, model, models_dir, league, kind):
    """Column list the model expects, verified against the dataframe."""
    features = load_manifest(models_dir, league, kind)
    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(
                f"{league}/{kind}: the model was trained on {len(features)} columns but "
                f"{len(missing)} of them are gone from the processed data "
                f"({', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}). "
                f"Re-run Download data, then Train models."
            )
        return features

    features = feature_columns(df)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != len(features):
        raise ValueError(
            f"{league}/{kind}: model expects {expected} features, processed data has "
            f"{len(features)}. The model predates the current data — re-run Train models."
        )
    return features


def resolve_features(df, model, models_dir, league, kind):
    """Return (X, features) for one prediction row."""
    features = resolve_feature_list(df, model, models_dir, league, kind)
    return df[features].values, features
