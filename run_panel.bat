@echo off
rem Start the control panel and open it in the default browser.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Virtual environment not found at %PY%
    echo Create it first, see RUNNING.md section 2.
    pause
    exit /b 1
)

rem Give the server a moment to bind before the browser asks for the page.
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:8000"

echo ============================================================
echo   AI Football Predictions - control panel
echo   http://127.0.0.1:8000
echo   Close this window to stop the server.
echo ============================================================
echo.
"%PY%" webapp\server.py
pause
