@echo off
REM ---------------------------------------------------------------------------
REM  ML Arena - double-click launcher for Windows
REM
REM  Uses the project virtual environment when one exists, otherwise falls back
REM  to whatever "python" is on PATH.
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"

if not defined PYEXE (
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo   Python was not found on this computer.
        echo   Install Python 3.11 or newer from https://www.python.org/downloads/
        echo   and tick "Add Python to PATH" during setup, then run this file again.
        echo.
        pause
        exit /b 1
    )
    set "PYEXE=python"
    echo   No .venv folder found - using the system Python.
    echo   If packages are missing, run:  pip install -r requirements.txt
    echo.
)

echo   Starting ML Arena...
echo.
"%PYEXE%" run.py %*

if errorlevel 1 (
    echo.
    echo   ML Arena stopped with an error. The message above explains why.
    pause
)

endlocal
