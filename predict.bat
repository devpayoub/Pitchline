@echo off
rem Fetch fresh fixtures (API leagues + Europa League) and regenerate final_predictions.txt
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at %PY%
    echo Create it first, see RUNNING.md section 2.
    exit /b 1
)

echo === Fetching Champions League and domestic fixtures ===
"%PY%" scripts\acquire_next_matches.py --get_teams_names_dir data\processed --next_matches_output_file data\next_matches.json
if errorlevel 1 goto :failed

echo.
echo === Fetching Europa League fixtures ===
"%PY%" scripts\fetch_europa_league.py --window_days 7
if errorlevel 1 goto :failed

echo.
echo === Generating predictions ===
"%PY%" scripts\make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data\processed --final_predictions_out_file final_predictions.txt --next_matches data\next_matches.json
if errorlevel 1 goto :failed

echo.
echo Done. Open final_predictions.txt
exit /b 0

:failed
echo.
echo [ERROR] A step failed. Read the message above.
exit /b 1
