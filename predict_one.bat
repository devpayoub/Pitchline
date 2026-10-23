@echo off
rem Predict one competition, or every competition.
rem   predict_one.bat            -> interactive menu
rem   predict_one.bat E0         -> direct, no menu (any letter case)
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at %PY%
    echo Create it first, see RUNNING.md section 2.
    pause
    exit /b 1
)

set "CODE="
if not "%~1"=="" for %%L in (CL EL E0 I1 D1 SP1 F1 ALL) do if /i "%~1"=="%%L" set "CODE=%%L"
if not "%~1"=="" if not defined CODE (
    echo Unknown league "%~1".
    echo Valid codes: CL EL E0 I1 D1 SP1 F1 ALL
    pause
    exit /b 1
)
if defined CODE goto :run

:menu
cls
echo ==================================================
echo    AI Football Predictions - pick a competition
echo ==================================================
echo    1^) Champions League      CL
echo    2^) Europa League         EL
echo    3^) Premier League        E0
echo    4^) Serie A               I1
echo    5^) Bundesliga            D1
echo    6^) La Liga               SP1
echo    7^) Ligue 1               F1
echo    8^) All competitions
echo    0^) Exit
echo.
set "CHOICE="
set /p "CHOICE=Enter a number and press Enter: "

if "%CHOICE%"=="1" set "CODE=CL"
if "%CHOICE%"=="2" set "CODE=EL"
if "%CHOICE%"=="3" set "CODE=E0"
if "%CHOICE%"=="4" set "CODE=I1"
if "%CHOICE%"=="5" set "CODE=D1"
if "%CHOICE%"=="6" set "CODE=SP1"
if "%CHOICE%"=="7" set "CODE=F1"
if "%CHOICE%"=="8" set "CODE=ALL"
if "%CHOICE%"=="0" exit /b 0

if not defined CODE (
    echo.
    echo "%CHOICE%" is not a valid choice.
    timeout /t 2 >nul
    goto :menu
)

:run
echo.
if "%CODE%"=="ALL" goto :run_all

echo === Predicting %CODE% ===
"%PY%" scripts\make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data\processed --final_predictions_out_file predictions_%CODE%.txt --next_matches data\next_matches.json --leagues %CODE%
if errorlevel 1 goto :failed
echo.
echo Done. Open predictions_%CODE%.txt
echo.
pause
exit /b 0

:run_all
echo === Predicting every competition ===
"%PY%" scripts\make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data\processed --final_predictions_out_file final_predictions.txt --next_matches data\next_matches.json
if errorlevel 1 goto :failed
echo.
echo Done. Open final_predictions.txt
echo.
pause
exit /b 0

:failed
echo.
echo [ERROR] Prediction failed. Read the message above.
echo Tip: run predict.bat first to refresh the fixture list.
pause
exit /b 1
