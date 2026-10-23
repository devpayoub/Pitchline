"""
Full pipeline: run full-market predictions then fetch live Flashscore odds and value-bet analysis.

Usage:
    python scripts/run_full_pipeline.py
"""
import os
import subprocess
import sys

PYTHON = sys.executable
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, cwd=BASE)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    # Step 1: Generate full-market predictions
    run(
        f'"{PYTHON}" scripts/make_predictions_full.py '
        f'--input_leagues_models_dir models '
        f'--input_data_predict_dir data/processed '
        f'--final_predictions_out_file data/final_predictions_full.txt '
        f'--next_matches data/next_matches.json'
    )

    # Step 2: Fetch live odds and generate value-bet report
    run(f'"{PYTHON}" scripts/fetch_flashscore_odds_v3.py')

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("Outputs:")
    print("  - data/final_predictions_full.txt  (full market predictions)")
    print("  - data/value_bets_report.md        (value bet analysis)")


if __name__ == "__main__":
    main()
