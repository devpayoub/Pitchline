"""
AI Football Predictions Script — Full Market Coverage

Predicts: 1X2, O/U 1.5, O/U 2.5, O/U 3.5, BTTS, Double Chance, DNB

Usage:
    python scripts/make_predictions_full.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file data/final_predictions_full.txt --next_matches data/next_matches.json
"""

import pandas as pd
import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as features_mod

# Define global constants
VALID_LEAGUES = ["CL", "EL", "E0", "I1", "D1", "SP1", "F1"]

# Competitions that reuse another league's model and processed data.
MODEL_ALIAS = {"EL": "CL"}

# Base-rate correction applied to European competitions.
PRIOR_SHIFT = {"CL": 2.30, "EL": 1.48}

# All target columns to drop from the feature matrix
ALL_TARGETS = {"Over2.5", "Result", "BTTS", "Over1.5", "Over3.5"}

# Features lists (kept for reference; actual features come from the processed data)
HOME_TEAM_FEATURES = [
    'HomeTeam', 'FTHG', 'HG', 'HTHG', 'HS', 'HST', 'HHW', 'HC', 'HF', 'HFKC', 'HO', 'HY', 'HR', 'HBP',
    'B365H', 'BFH', 'BSH', 'BWH', 'GBH', 'IWH', 'LBH', 'PSH', 'SOH', 'SBH', 'SJH', 'SYH', 'VCH', 'WHH',
    'BbMxH', 'BbAvH', 'MaxH', 'AvgH', 'BFEH', 'BbMxAHH', 'BbAvAHH', 'GBAHH', 'LBAHH', 'B365AHH', 'PAHH',
    'MaxAHH', 'AvgAHH', 'BbAHh', 'AHh', 'GBAH', 'LBAH', 'B365AH', 'AvgHomeGoalsScored', 'AvgHomeGoalsConceded',
    'HomeOver2.5Perc', 'AvgLast5HomeGoalsScored', 'AvgLast5HomeGoalsConceded', 'Last5HomeOver2.5Count', 'Last5HomeOver2.5Perc'
]

AWAY_TEAM_FEATURES = [
    'AwayTeam', 'FTAG', 'AG', 'HTAG', 'AS', 'AST', 'AHW', 'AC', 'AF', 'AFKC', 'AO', 'AY', 'AR', 'ABP',
    'B365A', 'BFA', 'BSA', 'BWA', 'GBA', 'IWA', 'LBA', 'PSA', 'SOA', 'SBA', 'SJA', 'SYA', 'VCA', 'WHA',
    'BbMxA', 'BbAvA', 'MaxA', 'AvgA', 'BFEA', 'BbMxAHA', 'BbAvAHA', 'GBAHA', 'LBAHA', 'B365AHA', 'PAHA',
    'MaxAHA', 'AvgAHA', 'AvgAwayGoalsScored', 'AvgAwayGoalsConceded', 'AwayOver2.5Perc', 'AvgLast5AwayGoalsScored',
    'AvgLast5AwayGoalsConceded', 'Last5AwayOver2.5Count', 'Last5AwayOver2.5Perc'
]


def load_model(filepath: str):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_league_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)


def prepare_row_to_predict(home_team_df: pd.DataFrame, away_team_df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    row_to_predict = pd.DataFrame(columns=numeric_columns)
    row_to_predict.loc[len(row_to_predict)] = [None] * len(row_to_predict.columns)

    # The most recent row already carries the shifted rolling/season features, i.e.
    # the team's state going into this fixture. head(5) used to grab the *oldest*
    # five rows, so predictions were made from the start of the dataset.
    home_team_final_df = home_team_df.tail(1)[numeric_columns]
    away_team_final_df = away_team_df.tail(1)[numeric_columns]

    for column in row_to_predict.columns:
        if column in HOME_TEAM_FEATURES:
            row_to_predict.loc[len(row_to_predict)-1, column] = home_team_final_df[column].mean()
        elif column in AWAY_TEAM_FEATURES:
            row_to_predict.loc[len(row_to_predict)-1, column] = away_team_final_df[column].mean()
        else:
            row_to_predict.loc[len(row_to_predict)-1, column] = (away_team_final_df[column].mean() + home_team_final_df[column].mean()) / 2

    return row_to_predict


def predict_binary(model, X_test, shift=None):
    """Return probability of class 1 for a binary model."""
    prob = model.predict_proba(X_test)[0]
    p = float(prob[1])
    if shift is not None:
        p = (p * shift) / (p * shift + (1 - p))
    return p


def predict_1x2(model, X_test):
    """Return dict of {home, draw, away} probabilities."""
    prob = model.predict_proba(X_test)[0]
    classes = list(model.classes_)
    idx = {cls: i for i, cls in enumerate(classes)}
    return {
        "home": float(prob[idx.get(0, 0)]),
        "draw": float(prob[idx.get(1, 1)]),
        "away": float(prob[idx.get(2, 2)]),
    }


def format_pct(p: float) -> str:
    return f"{p*100:.1f}%"


MODEL_KINDS = {
    "ou25": "voting_classifier",
    "1x2": "result_classifier",
    "btts": "btts_classifier",
    "ou15": "over15_classifier",
    "ou35": "over35_classifier",
}


def make_match_predictions(match, league: str, models: dict, league_data: pd.DataFrame,
                           models_dir: str = "models", model_league: str = None) -> dict:
    """Return a dict with all predictions for one match."""
    home_team = match['home_team']
    away_team = match['away_team']

    if home_team not in league_data['HomeTeam'].values or away_team not in league_data['AwayTeam'].values:
        return None

    home_team_df = league_data[league_data['HomeTeam'] == home_team]
    away_team_df = league_data[league_data['AwayTeam'] == away_team]

    model_league = model_league or league
    cache = {}

    def X_for(key, model):
        """Feature matrix for one model, built from its training manifest."""
        feats = features_mod.resolve_feature_list(
            league_data, model, models_dir, model_league, MODEL_KINDS[key])
        k = tuple(feats)
        if k not in cache:
            cache[k] = prepare_row_to_predict(home_team_df, away_team_df, feats).values
        return cache[k]

    result = {
        "home": home_team,
        "away": away_team,
        "date": match.get('date', ''),
    }

    # 1X2
    if models.get("1x2") is not None:
        p1x2 = predict_1x2(models["1x2"], X_for("1x2", models["1x2"]))
        result["p_home"] = p1x2["home"]
        result["p_draw"] = p1x2["draw"]
        result["p_away"] = p1x2["away"]
        # Derived markets
        result["dc_1x"] = p1x2["home"] + p1x2["draw"]
        result["dc_12"] = p1x2["home"] + p1x2["away"]
        result["dc_x2"] = p1x2["draw"] + p1x2["away"]
        # DNB (remove draw, renormalize)
        non_draw = p1x2["home"] + p1x2["away"]
        if non_draw > 0:
            result["dnb_home"] = p1x2["home"] / non_draw
            result["dnb_away"] = p1x2["away"] / non_draw
        else:
            result["dnb_home"] = 0.5
            result["dnb_away"] = 0.5

    # O/U 2.5
    if models.get("ou25") is not None:
        shift = PRIOR_SHIFT.get(league)
        result["p_over25"] = predict_binary(models["ou25"], X_for("ou25", models["ou25"]), shift)
        result["p_under25"] = 1 - result["p_over25"]

    # O/U 1.5
    if models.get("ou15") is not None:
        result["p_over15"] = predict_binary(models["ou15"], X_for("ou15", models["ou15"]))
        result["p_under15"] = 1 - result["p_over15"]

    # O/U 3.5
    if models.get("ou35") is not None:
        result["p_over35"] = predict_binary(models["ou35"], X_for("ou35", models["ou35"]))
        result["p_under35"] = 1 - result["p_over35"]

    # BTTS
    if models.get("btts") is not None:
        result["p_btts_yes"] = predict_binary(models["btts"], X_for("btts", models["btts"]))
        result["p_btts_no"] = 1 - result["p_btts_yes"]

    return result


def format_match_card(pred: dict) -> str:
    """Format a single match's predictions into a card."""
    home = pred["home"]
    away = pred["away"]
    lines = [f"⚽ **{home}** 🆚 **{away}**"]

    # 1X2
    if "p_home" in pred:
        lines.append(
            f"   🏆 1X2: {home} {format_pct(pred['p_home'])} | "
            f"Draw {format_pct(pred['p_draw'])} | {away} {format_pct(pred['p_away'])}"
        )

    # Goals markets
    goal_lines = []
    if "p_over15" in pred:
        pick15 = "Over" if pred["p_over15"] >= 0.5 else "Under"
        goal_lines.append(f"O/U 1.5: {pick15} {format_pct(max(pred['p_over15'], pred['p_under15']))}")
    if "p_over25" in pred:
        pick25 = "Over" if pred["p_over25"] >= 0.5 else "Under"
        goal_lines.append(f"O/U 2.5: {pick25} {format_pct(max(pred['p_over25'], pred['p_under25']))}")
    if "p_over35" in pred:
        pick35 = "Over" if pred["p_over35"] >= 0.5 else "Under"
        goal_lines.append(f"O/U 3.5: {pick35} {format_pct(max(pred['p_over35'], pred['p_under35']))}")
    if goal_lines:
        lines.append("   🥅 " + " | ".join(goal_lines))

    # BTTS
    if "p_btts_yes" in pred:
        pick = "Yes" if pred["p_btts_yes"] >= 0.5 else "No"
        lines.append(f"   🤝 BTTS: {pick} {format_pct(max(pred['p_btts_yes'], pred['p_btts_no']))}")

    # Derived markets
    derived = []
    if "dc_1x" in pred:
        derived.append(f"DC 1X {format_pct(pred['dc_1x'])}")
        derived.append(f"12 {format_pct(pred['dc_12'])}")
        derived.append(f"X2 {format_pct(pred['dc_x2'])}")
    if "dnb_home" in pred:
        dnb_pick = home if pred["dnb_home"] >= 0.5 else away
        derived.append(f"DNB {dnb_pick} {format_pct(max(pred['dnb_home'], pred['dnb_away']))}")
    if derived:
        lines.append("   🛡️ " + " | ".join(derived))

    lines.append("")
    return "\n".join(lines)


def make_predictions(league: str, models: dict, league_data: pd.DataFrame, competitions: dict,
                     models_dir: str = 'models', model_league: str = None) -> str:
    league_section = ""
    for competition_league, competitions_info in competitions.items():
        if competition_league == league:
            league_section = f"**{competitions_info['name']}**:\n\n"
            for match in competitions_info["next_matches"]:
                pred = make_match_predictions(match, league, models, league_data, models_dir, model_league)
                if pred is not None:
                    league_section += format_match_card(pred)
    return league_section


def main(input_leagues_models_dir: str, input_data_predict_dir: str, final_predictions_out_file: str, next_matches: str):
    try:
        print("Loading JSON file with upcoming matches...\n")
        with open(next_matches, 'r', encoding='utf-16') as json_file:
            competitions = json.load(json_file)
    except Exception as e:
        raise Exception(f"Error loading JSON file: {e}")

    predictions_message = (
        f"🎯 **AI Football Predictions: Full Market Coverage** 🎯\n\n"
        f"Markets: 1X2 | O/U 1.5 | O/U 2.5 | O/U 3.5 | BTTS | Double Chance | DNB\n"
        f"Predictions done: {datetime.now().strftime('%Y-%m-%d')}\n\n"
    )

    for league in VALID_LEAGUES:
        print(f"----------------------------------")
        print(f"\nMaking predictions for {league}...\n")
        model_league = MODEL_ALIAS.get(league, league)
        data_path = os.path.join(input_data_predict_dir, f"{model_league}_merged_preprocessed.csv")

        if not os.path.exists(data_path):
            print(f"Missing data for {league}. Skipping...")
            continue

        # Load all available models for this league
        models = {}

        # O/U 2.5
        ou25_path = os.path.join(input_leagues_models_dir, f"{model_league}_voting_classifier.pkl")
        if os.path.exists(ou25_path):
            models["ou25"] = load_model(ou25_path)

        # 1X2
        result_path = os.path.join(input_leagues_models_dir, f"{model_league}_result_classifier.pkl")
        if os.path.exists(result_path):
            models["1x2"] = load_model(result_path)

        # BTTS
        btts_path = os.path.join(input_leagues_models_dir, f"{model_league}_btts_classifier.pkl")
        if os.path.exists(btts_path):
            models["btts"] = load_model(btts_path)

        # O/U 1.5
        ou15_path = os.path.join(input_leagues_models_dir, f"{model_league}_over15_classifier.pkl")
        if os.path.exists(ou15_path):
            models["ou15"] = load_model(ou15_path)

        # O/U 3.5
        ou35_path = os.path.join(input_leagues_models_dir, f"{model_league}_over35_classifier.pkl")
        if os.path.exists(ou35_path):
            models["ou35"] = load_model(ou35_path)

        if not models:
            print(f"No models found for {league}. Skipping...")
            continue

        league_data = load_league_data(data_path)
        print(f"Loaded {len(models)} models for {league}.")
        print(f"Predicting matches for {league}...")
        league_section = make_predictions(league, models, league_data, competitions,
                                                  input_leagues_models_dir, model_league)
        print(f"Predictions made for {league}.")
        predictions_message += league_section + "\n"

    with open(final_predictions_out_file, 'w', encoding='utf-8') as file:
        file.write(predictions_message)
        print(f"\nPredictions saved to {final_predictions_out_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Football Predictions — Full Market Coverage")
    parser.add_argument('--input_leagues_models_dir', type=str, required=True)
    parser.add_argument('--input_data_predict_dir', type=str, required=True)
    parser.add_argument('--final_predictions_out_file', type=str, required=True)
    parser.add_argument('--next_matches', type=str, required=True)
    parser.add_argument('--leagues', nargs='+', default=None,
                        help="Only predict these competition codes, e.g. --leagues E0")

    args = parser.parse_args()
    if args.leagues:
        requested = [code.upper() for code in args.leagues]
        unknown = [code for code in requested if code not in VALID_LEAGUES]
        if unknown:
            raise SystemExit(f"Unknown league code(s): {', '.join(unknown)}. Supported: {', '.join(VALID_LEAGUES)}")
        VALID_LEAGUES[:] = [code for code in VALID_LEAGUES if code in requested]
    main(args.input_leagues_models_dir, args.input_data_predict_dir, args.final_predictions_out_file, args.next_matches)
