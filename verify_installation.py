"""
Verification script for Python installation and project dependencies.
Run this after setting up Python and installing requirements.
"""

import sys
import os
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def check_python_version():
    """Check Python version."""
    print_section("Python Version Check")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"Installation path: {sys.executable}")
    
    if version.major == 3 and version.minor >= 10:
        print("✅ Python version is compatible (3.10+)")
        return True
    else:
        print("❌ Python version is NOT compatible (requires 3.10+)")
        return False


def check_installation_location():
    """Check if Python is on E: drive."""
    print_section("Installation Location Check")
    python_path = Path(sys.executable)
    print(f"Python installed at: {python_path}")
    
    if python_path.drive == 'E:':
        print("✅ Python is installed on E: drive")
        return True
    elif python_path.drive == 'C:':
        print("⚠️  Python is still on C: drive")
        print("   Consider reinstalling to E: drive to save space")
        return False
    else:
        print(f"ℹ️  Python is on {python_path.drive} drive")
        return True


def check_virtual_environment():
    """Check if running in a virtual environment."""
    print_section("Virtual Environment Check")
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✅ Running in a virtual environment")
        print(f"   Virtual env path: {sys.prefix}")
        return True
    else:
        print("⚠️  NOT running in a virtual environment")
        print("   Recommended: Create and activate a venv for this project")
        return False


def check_core_packages():
    """Check if core packages are installed."""
    print_section("Core Package Check")
    packages = {
        'torch': 'PyTorch (required for GRU training)',
        'numpy': 'NumPy (required for training)',
        'pandas': 'Pandas (data handling)',
        'sklearn': 'scikit-learn (metrics)',
        'matplotlib': 'Matplotlib (visualization)',
    }
    
    all_installed = True
    for package, description in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package:12} v{version:10} - {description}")
        except ImportError:
            print(f"❌ {package:12} {'NOT FOUND':10} - {description}")
            all_installed = False
    
    return all_installed


def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print_section("PyTorch CUDA Check")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"GPU 0: {torch.cuda.get_device_name(0)}")
            print("✅ PyTorch CUDA is properly configured")
            return True
        else:
            print("⚠️  CUDA is NOT available (CPU-only PyTorch)")
            print("   If you have a GPU, reinstall PyTorch with CUDA support:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False


def check_project_structure():
    """Check if project files exist."""
    print_section("Project Structure Check")
    required_files = [
        'main.py',
        'evaluate.py',
        'requirements.txt',
        'hybrid_reasoner/__init__.py',
        'dataset/test_prompts.json',
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_ollama():
    """Check if Ollama is accessible."""
    print_section("Ollama Connection Check")
    try:
        import urllib.request
        import json
        
        url = "http://localhost:11434/api/version"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                version = data.get('version', 'unknown')
                print(f"✅ Ollama is running (version: {version})")
                print("   Checking for phi3 model...")
                
                # Check for phi3 model
                models_url = "http://localhost:11434/api/tags"
                with urllib.request.urlopen(models_url, timeout=5) as models_response:
                    models_data = json.loads(models_response.read())
                    models = models_data.get('models', [])
                    phi3_found = any('phi3' in model.get('name', '').lower() for model in models)
                    
                    if phi3_found:
                        print("✅ phi3 model is installed")
                    else:
                        print("⚠️  phi3 model NOT found")
                        print("   Install with: ollama pull phi3")
                
                return phi3_found
        except urllib.error.URLError:
            print("❌ Cannot connect to Ollama (is it running?)")
            print("   Start Ollama with: ollama serve")
            print("   Then install phi3: ollama pull phi3")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("  PYTHON INSTALLATION VERIFICATION")
    print("  Project: Hybrid Reasoning System")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results['python_version'] = check_python_version()
    results['install_location'] = check_installation_location()
    results['virtual_env'] = check_virtual_environment()
    results['core_packages'] = check_core_packages()
    results['pytorch_cuda'] = check_pytorch_cuda()
    results['project_structure'] = check_project_structure()
    results['ollama'] = check_ollama()
    
    # Summary
    print_section("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    print()
    
    critical_checks = ['python_version', 'core_packages', 'project_structure']
    critical_failed = any(not results[check] for check in critical_checks if check in results)
    
    if critical_failed:
        print("❌ CRITICAL CHECKS FAILED")
        print("   Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    elif passed == total:
        print("✅ ALL CHECKS PASSED!")
        print("   Your environment is ready to use.")
        print()
        print("   Try running:")
        print("   python main.py \"Tom has 3 apples.\" --scenario apple_sharing")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("   Review the warnings above.")
        print("   The system may still work with limited functionality.")
        return 0


if __name__ == "__main__":
    sys.exit(main())




