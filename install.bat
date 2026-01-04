@echo off
REM Emotion Symphony - Windows Installation Script

echo ==========================================
echo    EMOTION SYMPHONY - Installation
echo ==========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found! Please install Python 3.8+ first.
    echo     Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [X] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded
echo.

REM Install dependencies
echo Installing Python dependencies...
echo This may take a few minutes...
cd python
pip install -r requirements.txt --break-system-packages
if %errorlevel% neq 0 (
    echo [!] Some dependencies may have failed to install
    echo     Check the error messages above
) else (
    echo [OK] Dependencies installed successfully
)
cd ..

echo.
echo ==========================================
echo    Installation Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Quick Start (Web App):
echo    - Open web\index.html in your browser
echo.
echo 2. Run Music Demo:
echo    - cd python
echo    - python demo.py
echo.
echo 3. Train Model (requires dataset):
echo    - Download FER-2013 from Kaggle
echo    - python emotion_model.py train ..\data\fer2013.csv
echo.
echo 4. Check SETUP.md for detailed instructions
echo.
echo Happy coding! [Music Note]
echo.
pause
