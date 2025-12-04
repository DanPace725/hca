@echo off
REM Quick installation for CUDA 11.8 GPU setup

echo ==========================================
echo Installing Hybrid Reasoner - GPU Setup
echo ==========================================
echo.

echo Step 1: Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch installation failed
    pause
    exit /b 1
)

echo.
echo Step 2: Installing other dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Dependencies installation failed
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installation complete!
echo ==========================================
echo.
echo Test your installation:
echo   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.
pause

