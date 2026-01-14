@echo off
REM Quick start script for MRD Detection API (Windows)

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo MRD Detection API - Quick Start (Windows)
echo ==========================================
echo.

REM Check if model file exists
if not exist "model\vae_4dim_6_final.pth" (
    echo ERROR: Model file not found: model\vae_4dim_6_final.pth
    exit /b 1
)

echo [OK] Model file found
echo.

REM Create Python virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo.
echo ==========================================
echo [OK] Setup complete!
echo ==========================================
echo.
echo To start the API, run:
echo   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
echo.
echo API will be available at:
echo   - http://localhost:8000
echo   - API docs: http://localhost:8000/docs
echo.
echo To test the API, run in another terminal:
echo   python test_client.py
echo.
echo Or use Docker:
echo   docker-compose up -d
echo.
pause
