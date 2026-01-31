@echo off
setlocal

cd /d "%~dp0"

echo === HeartMuse Music Generator - Installation ===

where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

python --version

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Upgrading pip...
pip install --upgrade pip

if not exist "heartlib" (
    echo Cloning heartlib...
    git clone https://github.com/HeartMuLa/heartlib.git
)

echo Installing heartlib...
pip install -e heartlib

echo Installing app dependencies...
pip install -r requirements.txt

echo.
echo === Installation complete! ===
echo Run run.bat to start the application.
pause
