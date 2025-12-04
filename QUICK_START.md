# Quick Start After Python Reinstall

## Installation Summary

**Target**: Python 3.11 on `E:\Python\Python311\`  
**Project**: `E:\Coding\hca\hca`  
**Virtual Environment**: `E:\Coding\hca\hca\venv`

---

## Daily Workflow

### Starting a Work Session

```powershell
# 1. Open PowerShell/Terminal
# 2. Navigate to project
cd E:\Coding\hca\hca

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1
# You should see (venv) in your prompt

# 4. Start working!
python main.py "Your prompt here" --scenario apple_sharing
```

### Ending a Work Session

```powershell
# Just close the terminal or type:
deactivate
```

---

## Common Commands

```powershell
# Check Python version
python --version

# Run main program
python main.py "Tom has 3 apples." --scenario apple_sharing

# Run evaluation
python evaluate.py --tasks dataset/test_prompts.json --limit 10

# Run verification script
python verify_installation.py

# Install/update a package
pip install package-name

# List installed packages
pip list

# Reinstall requirements
pip install -r requirements.txt
```

---

## Available Scenarios

- `apple_sharing` - Apple distribution problems
- `cookie_problem` - Cookie sharing
- `travel_distance` - Travel calculations  
- `recipe_scaling` - Recipe scaling

---

## Troubleshooting

### "Command not found" or "Python not recognized"

**Problem**: Terminal can't find Python  
**Solution**: 
1. Close ALL terminal windows
2. Open a NEW terminal
3. Check PATH: `where.exe python`
4. Should show: `E:\Python\Python311\python.exe`

### Virtual environment won't activate

**Problem**: Script execution policy  
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Module not found" errors

**Problem**: Dependencies not installed or wrong Python  
**Solution**:
```powershell
# Make sure you're in the virtual environment (should see (venv))
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### CUDA not available

**Problem**: PyTorch can't find GPU  
**Solution**:
```powershell
# Check if NVIDIA drivers are working
nvidia-smi

# Reinstall PyTorch with CUDA
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Ollama connection failed

**Problem**: Can't connect to Ollama service  
**Solution**:
```powershell
# Start Ollama service (in a separate terminal)
ollama serve

# Pull phi3 model if needed
ollama pull phi3

# Test connection
curl http://localhost:11434/api/version
```

---

## File Locations

| What | Where |
|------|-------|
| Python Installation | `E:\Python\Python311\` |
| Project Root | `E:\Coding\hca\hca\` |
| Virtual Environment | `E:\Coding\hca\hca\venv\` |
| Installed Packages | `E:\Coding\hca\hca\venv\Lib\site-packages\` |
| Trace Logs | `E:\Coding\hca\hca\runs\` |
| Test Data | `E:\Coding\hca\hca\dataset\` |

---

## Installation Steps (Reference)

1. ✅ Uninstall Python 3.12 and 3.13 from C: drive
2. ✅ Download Python 3.11.10 from python.org
3. ✅ Install to `E:\Python\Python311\` with custom installation
4. ✅ Check "Add to PATH" during installation
5. ✅ Run `setup_after_python_install.bat` in project folder
6. ✅ Verify with `python verify_installation.py`
7. ✅ Start working!

---

## Need Help?

- Full guide: See `cleanup_and_reinstall_python.md`
- Verify setup: Run `python verify_installation.py`
- Project docs: See `README.md` and `INSTALL.md`
- Check traces: Look in `runs/` folder for execution logs

---

## Remember

- ✅ **ALWAYS** activate the virtual environment before working
- ✅ Your prompt should show `(venv)` when activated
- ✅ Python is now on E: drive - saves C: drive space
- ✅ Virtual environment keeps project dependencies isolated
- ✅ Run verification script after any major changes




