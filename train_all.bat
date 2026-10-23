@echo off
rem Run the whole pipeline: download data, preprocess, train both markets, predict.
rem This takes roughly 30-40 minutes on a 12-core machine.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at %PY%
    exit /b 1
)
set OMP_NUM_THREADS=1

echo === 1/6 Downloading league data ===
"%PY%" scripts\data_acquisition.py --leagues E0 I1 SP1 F1 D1 --seasons 2627 2526 2425 2324 --raw_data_output_dir data\raw
if errorlevel 1 goto :failed

echo.
echo === 2/6 Downloading extra European leagues (for the CL model) ===
"%PY%" scripts\data_acquisition.py --leagues N1 B1 P1 T1 G1 --seasons 2627 2526 2425 2324 --raw_data_output_dir data\raw
if errorlevel 1 goto :failed

echo.
echo === 3/6 Pooling ten leagues into the European dataset ===
"%PY%" scripts\build_european_dataset.py
if errorlevel 1 goto :failed

echo.
echo === 4/6 Preprocessing ===
"%PY%" scripts\data_preprocessing.py --raw_data_input_dir data\raw --processed_data_output_dir data\processed --num_features 20 --clustering_threshold 0.5 --target over25
if errorlevel 1 goto :failed

echo.
echo === 5/6 Training over/under models ===
"%PY%" scripts\train_models.py --processed_data_input_dir data\processed --trained_models_output_dir models --metric_choice accuracy --n_splits 5 --voting soft --target over25
if errorlevel 1 goto :failed

echo.
echo === 5b/6 Training 1X2 models ===
"%PY%" scripts\train_models.py --processed_data_input_dir data\processed --trained_models_output_dir models --metric_choice accuracy --n_splits 3 --voting soft --target result
if errorlevel 1 goto :failed

echo.
echo === 6/6 Fetching fixtures and predicting ===
"%PY%" scripts\acquire_next_matches.py --get_teams_names_dir data\processed --next_matches_output_file data\next_matches.json
if errorlevel 1 goto :failed
"%PY%" scripts\fetch_europa_league.py --window_days 7
if errorlevel 1 goto :failed
"%PY%" scripts\make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data\processed --final_predictions_out_file final_predictions.txt --next_matches data\next_matches.json
if errorlevel 1 goto :failed

echo.
echo All done. Open final_predictions.txt
exit /b 0

:failed
echo.
echo [ERROR] A step failed. Read the message above.
exit /b 1
