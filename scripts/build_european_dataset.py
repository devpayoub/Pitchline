"""
Build a pooled "European" dataset used to predict Champions League fixtures.

Why this exists
---------------
football-data.co.uk only publishes domestic league files, there is no Champions
League results file (CL.csv / UCL.csv return 404). The football-data.org API does
serve Champions League fixtures, but it exposes no shots, corners or bookmaker
odds, which are exactly the inputs the feature engineering needs.

So instead of training on Champions League matches, we pool the five domestic
leagues we do have into a single dataset and let one model learn the generic
relationship "given both teams' recent scoring and conceding form, will this match
go over 2.5 goals?". Champions League fixtures are then scored with the recent
domestic form of the two clubs involved.

Output: data/raw/CL_merged.csv, which data_preprocessing.py turns into
data/processed/CL_merged_preprocessed.csv exactly like the other leagues.
"""
import os
import argparse
import pandas as pd

SOURCE_LEAGUES = ["E0", "I1", "SP1", "F1", "D1", "N1", "B1", "P1", "T1", "G1"]


def build_pooled_dataset(raw_data_dir: str, output_file: str) -> None:
    frames = []
    common_columns = None
    column_order = []

    for league in SOURCE_LEAGUES:
        path = os.path.join(raw_data_dir, f"{league}_merged.csv")
        if not os.path.exists(path):
            print(f"Skipping {league}: {path} not found")
            continue

        df = pd.read_csv(path)
        if common_columns is None:
            common_columns = set(df.columns)
            column_order = df.columns.tolist()
        else:
            common_columns.intersection_update(df.columns)

        frames.append(df)
        print(f"Loaded {league}: {len(df)} matches")

    if not frames:
        raise RuntimeError("No raw league files found, run data_acquisition.py first.")

    common_columns = [c for c in column_order if c in common_columns]
    pooled = pd.concat([df[common_columns] for df in frames], ignore_index=True)

    # The preprocessing step groups form by ['Season', 'Team'], and team names are
    # unique across the five leagues, so pooling does not mix clubs together.
    pooled = pooled.sort_values(by=["Date"]).reset_index(drop=True)
    pooled.to_csv(output_file, index=False)
    print(f"Saved pooled dataset with {len(pooled)} matches and {len(common_columns)} columns to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pool the five domestic leagues into one dataset for Champions League predictions.")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw", help="Directory holding <league>_merged.csv files")
    parser.add_argument("--output_file", type=str, default="data/raw/CL_merged.csv", help="Where to write the pooled dataset")
    args = parser.parse_args()

    build_pooled_dataset(args.raw_data_dir, args.output_file)
