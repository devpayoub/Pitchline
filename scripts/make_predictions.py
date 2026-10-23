"""
AI Football Predictions Script: Predicts the likelihood of over 2.5 goals in upcoming football matches.

This script loads pre-trained machine learning models and match data to predict whether upcoming football matches will end with more than 2.5 goals. The predictions are then formatted into a Telegram-ready message that can be shared directly. 

How to run:
1. Ensure the necessary data and model files are in the specified directories.
2. Run the script with the appropriate arguments to generate predictions.
-----------------------------------------------------------------------------

Example usage, it is suggested to run the script in the root directory:

    python scripts/make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file data/final_predictions.txt --next_matches data/next_matches.json

Required Libraries:
- pandas
- numpy
- pickle
- argparse
- datetime
- json
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

# Competitions that reuse another league's model and processed data. The Europa League
# runs on the same pooled European model as the Champions League: there is no EL results
# source on football-data.co.uk and the free API tier blocks competition 2143.
MODEL_ALIAS = {"EL": "CL"}

# Base-rate correction applied to European competitions. The pooled model is trained on
# domestic matches, which are tighter than European nights. Derived from the odds ratio of
# the observed over-2.5 rates against the domestic 54.4%:
#   CL 73.3% -> 2.30   |   EL 63.9% (36 matches, 2026-27) -> 1.48
PRIOR_SHIFT = {"CL": 2.30, "EL": 1.48}

# Every target column that lives in the processed CSVs and must never enter the feature matrix.
ALL_TARGETS = ("Over2.5", "Result", "BTTS", "Over1.5", "Over3.5")

# Define the features for home team, away team, and general match information
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

"""
The general features are common to both home and away teams and contain match information that is not specific to either team.
This list in no longer necessary because in case that a feature is not in the home or away team features, it will be considered as a general feature.
GENERAL_FEATURES = [
    'Div', 'Date', 'Time', 'FTR', 'Res', 'HTR', 'Attendance', 'Referee', 'Bb1X2', 'BbMxD', 'BbAvD', 'MaxD', 'AvgD',
    'B365D', 'BFD', 'BSD', 'BWD', 'GBD', 'IWD', 'LBD', 'PSD', 'SOD', 'SBD', 'SJD', 'SYD', 'VCD', 'WHD', 'BbOU',
    'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5',
    'Max>2.5', 'Max<2.5', 'Avg>2.5', 'AvgC>2.5', 'Avg<2.5', 'AvgC<2.5', 'MaxCAHA', 'MaxC>2.5', 'B365C<2.5', 'MaxCA',
    'B365CAHH', 'BbAH', 'Over2.5'
]
"""

def load_model(filepath: str):
    """Loads the machine learning model from a specified pickle file.
    
    Args:
        filepath (str): Path to the pickle file containing the model.
    
    Returns:
        model: The loaded machine learning model.
    """
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_league_data(filepath: str) -> pd.DataFrame:
    """Loads the league data from a CSV file using pandas.
    
    Args:
        filepath (str): Path to the CSV file containing league data.
    
    Returns:
        pd.DataFrame: The loaded league data as a DataFrame.
    """
    # check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    else:
        print(f"Loading data from {filepath}...")
    # Load the data from the CSV file
        return pd.read_csv(filepath)


def prepare_row_to_predict(home_team_df: pd.DataFrame, away_team_df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """Prepares a DataFrame row for prediction by averaging relevant team statistics.
    
    Args:
        home_team_df (pd.DataFrame): DataFrame containing the home team's data.
        away_team_df (pd.DataFrame): DataFrame containing the away team's data.
        numeric_columns (list): List of numeric columns for prediction.
    
    Returns:
        pd.DataFrame: A single row DataFrame ready for prediction.
    """
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
        # If the column is not in the home or away team features, we take the average of both teams
        else:
            row_to_predict.loc[len(row_to_predict)-1, column] = (away_team_final_df[column].mean() + home_team_final_df[column].mean()) / 2

    return row_to_predict


def make_predictions(league: str, league_model, league_data: pd.DataFrame, competitions: dict, league_result_model=None, models_dir: str = 'models', model_league: str = None) -> str:
    """Makes predictions for a specific league and formats them into a Telegram message.
    
    Args:
        league (str): The league identifier.
        league_model: The machine learning model for the league.
        league_data (pd.DataFrame): DataFrame containing the league data.
        competitions (dict): Dictionary containing competition details and upcoming matches.
        league_result_model: Optional 1X2 model. When provided, a second line with the
            home/draw/away probabilities is added under each match.
    
    Returns:
        str: A formatted string containing the predictions for the league.
    """
    model_league = model_league or league
    league_section = ""
    for competition_league, competitions_info in competitions.items():
        if competition_league == league:
            league_section = f"**{competitions_info['name']}**:\n"
            for match in competitions_info["next_matches"]:
                home_team = match['home_team']
                away_team = match['away_team']

                if home_team not in league_data['HomeTeam'].values or away_team not in league_data['AwayTeam'].values:
                    continue

                home_team_df = league_data[league_data['HomeTeam'] == home_team]
                away_team_df = league_data[league_data['AwayTeam'] == away_team]

                # Column list comes from the model's training manifest, so training and
                # prediction can never drift apart again.
                numeric_columns = features_mod.resolve_feature_list(
                    league_data, league_model, models_dir, model_league, "voting_classifier")

                row_to_predict = prepare_row_to_predict(home_team_df, away_team_df, numeric_columns)
                X_test = row_to_predict.values
                prediction = league_model.predict(X_test)
                predicted_probability = league_model.predict_proba(X_test)[0]
                p_over = float(predicted_probability[1])

                if league in PRIOR_SHIFT:
                    # The European models are trained on pooled domestic matches, which are
                    # much tighter than European nights: 2026-27 CL averaged 4.16 goals with
                    # 73.3% over 2.5 and EL 2.86 with 63.9%, against 2.84 goals and 54.4%
                    # domestically. Shift by the ratio of those base-rate odds so the output
                    # is not systematically biased towards "Under".
                    shift = PRIOR_SHIFT[league]
                    p_over = (p_over * shift) / (p_over * shift + (1 - p_over))

                if p_over >= 0.5:
                    result = f"Over 2.5 Goals! 🔥 ({round(p_over * 100, 2)}% chance)"
                else:
                    result = f"Under 2.5 Goals ({round((1 - p_over) * 100, 2)}% chance)"

                league_section += f"- ⚽ **{home_team}** 🆚 **{away_team}**: {result}\n"

                if league_result_model is not None:
                    # 1X2: classes are 0 = home win, 1 = draw, 2 = away win
                    result_proba = league_result_model.predict_proba(X_test)[0]
                    class_index = {cls: i for i, cls in enumerate(league_result_model.classes_)}
                    p_home = result_proba[class_index.get(0, 0)]
                    p_draw = result_proba[class_index.get(1, 1)]
                    p_away = result_proba[class_index.get(2, 2)]
                    league_section += (
                        f"   └ 1X2: {home_team} {round(p_home * 100, 1)}% | "
                        f"Draw {round(p_draw * 100, 1)}% | {away_team} {round(p_away * 100, 1)}%\n"
                    )

    return league_section


def main(input_leagues_models_dir: str, input_data_predict_dir: str, final_predictions_out_file: str, next_matches: str):
    """Main function that handles the entire prediction process.
    
    Args:
        input_leagues_models_dir (str): Directory containing the model files.
        input_data_predict_dir (str): Directory containing the processed data files.
        final_predictions_out_file (str): Path where the output Telegram message will be saved.
        next_matches (str): Path to the JSON file with upcoming matches information.
    """
    try:
        print("Loading JSON file with upcoming matches...\n")
        with open(next_matches, 'r', encoding='utf-16') as json_file:
            competitions = json.load(json_file)
    except Exception as e:
        raise Exception(f"Error loading JSON file: {e}")

    predictions_message = f"🎯 **AI Football Predictions: Over/Under 2.5 & 1X2** 🎯\n\nCheck out the latest predictions for the upcoming football matches! We've analyzed the data and here are our thoughts:\n PREDICTIONS DONE: {datetime.now().strftime('%Y-%m-%d')} \n\n"

    for league in VALID_LEAGUES:
        print(f"----------------------------------")
        print(f"\nMaking predictions for {league}...\n")
        model_league = MODEL_ALIAS.get(league, league)
        model_path = os.path.join(input_leagues_models_dir, f"{model_league}_voting_classifier.pkl")
        data_path = os.path.join(input_data_predict_dir, f"{model_league}_merged_preprocessed.csv")

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"Missing data or model for {league}. Skipping...")
            continue

        league_model = load_model(model_path)

        # The 1X2 model is optional: when it is missing the report just omits that line
        result_model_path = os.path.join(input_leagues_models_dir, f"{model_league}_result_classifier.pkl")
        league_result_model = load_model(result_model_path) if os.path.exists(result_model_path) else None

        league_data = load_league_data(data_path)
        print(f"Loaded model and data for {league}.")
        print(f"Predicting matches for {league}...")
        league_section = make_predictions(league, league_model, league_data, competitions, league_result_model,
                                             input_leagues_models_dir, model_league)
        print(f"Predictions made for {league}.")
        predictions_message += league_section + "\n"

    with open(final_predictions_out_file, 'w', encoding='utf-8') as file:
        file.write(predictions_message)
        print(f"\n Predictions saved to {final_predictions_out_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Football Predictions Script")
    parser.add_argument('--input_leagues_models_dir', type=str, required=True, help="Directory containing the model files")
    parser.add_argument('--input_data_predict_dir', type=str, required=True, help="Directory containing the processed data files")
    parser.add_argument('--final_predictions_out_file', type=str, required=True, help="File path to save the Telegram message output")
    parser.add_argument('--next_matches', type=str, required=True, help="Path to the JSON file with upcoming matches information")
    parser.add_argument('--leagues', nargs='+', default=None,
                        help="Only predict these competition codes, e.g. --leagues E0. Default: every supported league.")

    args = parser.parse_args()
    if args.leagues:
        requested = [code.upper() for code in args.leagues]
        unknown = [code for code in requested if code not in VALID_LEAGUES]
        if unknown:
            raise SystemExit(f"Unknown league code(s): {', '.join(unknown)}. Supported: {', '.join(VALID_LEAGUES)}")
        VALID_LEAGUES[:] = [code for code in VALID_LEAGUES if code in requested]
    main(args.input_leagues_models_dir, args.input_data_predict_dir, args.final_predictions_out_file, args.next_matches)
