@echo off
REM Setup script for Hybrid Reasoning System (Windows)

echo ==========================================
echo Hybrid Reasoning System - Setup
echo ==========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo [OK] Python detected
echo.

REM Ask about installation type
echo Select installation type:
echo 1) Minimal (rules + stub only, no dependencies)
echo 2) Full (with PyTorch GRU and training capabilities)
set /p choice="Enter choice [1-2]: "

if "%choice%"=="1" (
    echo.
    echo [OK] Minimal installation selected
    echo     No dependencies needed - uses Python stdlib only
    goto :done
)

if "%choice%"=="2" (
    echo.
    echo Installing dependencies...
    
    REM Check for CUDA
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo No GPU detected. Installing CPU-only PyTorch...
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install -r requirements.txt
    ) else (
        echo GPU detected. Install PyTorch with CUDA support? [y/N]
        set /p cuda_choice="> "
        if /i "%cuda_choice%"=="y" (
            echo Installing PyTorch with CUDA 12.1...
            pip install torch --index-url https://download.pytorch.org/whl/cu121
            pip install -r requirements.txt
        ) else (
            echo Installing CPU-only PyTorch...
            pip install torch --index-url https://download.pytorch.org/whl/cpu
            pip install -r requirements.txt
        )
    )
    echo [OK] Dependencies installed
    goto :done
)

echo Invalid choice
exit /b 1

:done
echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Install Ollama: https://ollama.ai
echo 2. Pull Phi-3 model: ollama pull phi3
echo 3. Start Ollama: ollama serve
echo 4. Run example: python main.py "Tom has 3 apples..." --scenario apple_sharing
echo.
echo See README.md for more information.
pause

