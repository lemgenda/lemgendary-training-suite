# training/lemgendary_hub.ps1
# Master Orchestration Script for LemGendary AI Training & Management

$script:HUB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $script:HUB_DIR) { $script:HUB_DIR = $PSScriptRoot }
if (-not $script:HUB_DIR) { $script:HUB_DIR = Get-Location }

# PowerShell 5.1 compatibility for Join-Path
$PARENT_DIR = Split-Path -Parent $script:HUB_DIR
$VENV_ROOT = Join-Path $PARENT_DIR ".venv"
$VENV_LOCAL = Join-Path $script:HUB_DIR ".venv"

# Default to local if neither exists
$script:VENV_DIR = if (Test-Path $VENV_ROOT) { $VENV_ROOT } else { $VENV_LOCAL }
$script:REQ_FILE = Join-Path $script:HUB_DIR "requirements.txt"

function Write-Header($text) {
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor White
    Write-Host "================================================================================`n" -ForegroundColor Cyan
}

function Test-Environment {
    if (-not (Test-Path "$script:VENV_DIR\Scripts\python.exe")) {
        Write-Host "  [!] Virtual environment not detected at $script:VENV_DIR" -ForegroundColor Yellow
        $choice = Read-Host "  Would you like to create it in the training folder? (Y/N)"
        if ($choice -eq 'Y' -or $choice -eq 'y') {
            $script:VENV_DIR = $VENV_LOCAL
            Initialize-Environment
            return $true
        }
        return $false
    }
    return $true
}

function Initialize-Environment {
    Write-Header "PREPARING PYTHON ENVIRONMENT"
    
    # Check for Python, install if missing
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "  [!] Python is not installed or not in PATH." -ForegroundColor Yellow
        Write-Host "  [+] Attempting silent autonomous installation via winget..." -ForegroundColor Cyan
        try {
            winget install --id Python.Python.3.10 -e --silent --accept-package-agreements --accept-source-agreements
            $env:Path += ";C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python310"
        } catch {
            Write-Host "  [ERROR] Auto-install failed. Please install Python 3.10 manually." -ForegroundColor Red
            return
        }
    }

    Write-Host "  [1/4] Creating virtual environment ($script:VENV_DIR)..." -ForegroundColor Cyan
    python -m venv $script:VENV_DIR

    Write-Host "  [2/4] Upgrading pip..." -ForegroundColor Cyan
    & "$script:VENV_DIR\Scripts\python.exe" -m pip install --upgrade pip

    Write-Host "  [3/4] Installing CUDA Core (NVIDIA Support)..." -ForegroundColor Cyan
    & "$script:VENV_DIR\Scripts\python.exe" -m pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

    Write-Host "  [4/4] Installing Auxiliary libraries (AMD/Intel/ONNX)..." -ForegroundColor Cyan
    & "$script:VENV_DIR\Scripts\python.exe" -m pip install -r $script:REQ_FILE

    Write-Host "`n  [SUCCESS] Environment is ready! You can now use all LemGendary tools." -ForegroundColor Green
}

function Show-Menu {
    Clear-Host
    Write-Header "LEMGENDARY MASTER HUB & ENVIRONMENT SUITE"
    Write-Host "  1. Initialize/Fix Environment  (Installs Python 3.10 and PyTorch 2.4.1)"
    Write-Host "  2. Train Individual Model      (Launches LemGendary Training Suite)"
    Write-Host "  3. Global Orchestration        (Automated sequential multi-model run)"
    Write-Host "  4. Deploy to Kaggle Cloud      (Generate Cloud Instructions)"
    Write-Host "  5. Smart Cloud Orchestration   (Local GPU + Dynamic Kaggle Streams)"
    Write-Host "  6. Single-Epoch Unit Test      (Diagnostic 1-Epoch pass for ALL models)"
    Write-Host "  7. Exit"
    Write-Host ""
}

while ($true) {
    Show-Menu
    $choice = Read-Host "Select an option (1-7)"

    switch ($choice) {
        '1' {
            Initialize-Environment
            Read-Host "`nPress Enter to return to menu..."
        }
        '2' {
            Write-Header "TRAIN INDIVIDUAL MODEL"
            Write-Host "  [DESCRIPTION]" -ForegroundColor White
            Write-Host "  Runs the LemGendary Unified Training Suite individually."
            Write-Host ""
            if (Test-Environment) {
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" "training/train.py"
                Pop-Location
            }
            Read-Host "`nPress Enter to return to menu..."
        }
        '3' {
            Write-Header "GLOBAL ORCHESTRATION"
            Write-Host "  [DESCRIPTION]" -ForegroundColor White
            Write-Host "  This initiates the massive chronological sequence executing"
            Write-Host "  all 21 models directly on your hardware flawlessly."
            Write-Host ""
            if (Test-Environment) {
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" "train_all.py"
                Pop-Location
            }
            Read-Host "`nPress Enter to return to menu..."
        }
        '4' {
            Write-Header "DEPLOY TO KAGGLE CLOUD"
            Write-Host "  [DESCRIPTION]" -ForegroundColor White
            Write-Host "  The LemGendary Neural Architecture is 100% prepared for native"
            Write-Host "  Cloud-GPU training on Kaggle. Python Jupyter Notebooks are"
            Write-Host "  already explicitly compiled in your root directory."
            Write-Host ""
            Write-Host "  [INSTRUCTIONS]" -ForegroundColor Cyan
            Write-Host "  1. Open Kaggle -> Create Notebook -> File -> Import Notebook."
            Write-Host "  2. Select one of the 'Kaggle_Train_Solo...' or 'Kaggle_Train_Multi...' files."
            Write-Host "  3. Click 'Add Data' in Kaggle and mount the explicitly requested datasets."
            Write-Host "  4. Go to 'Session Options' and set Accelerator to 'GPU T4 x2' or 'P100'."
            Write-Host "  5. Click 'Run All'."
            Write-Host ""
            Read-Host "Press Enter to return to menu..."
        }
        '5' {
            Write-Header "SMART CLOUD ORCHESTRATION"
            Write-Host "  [DESCRIPTION]" -ForegroundColor White
            Write-Host "  This mathematically optimizes PyTorch training by automatically downloading"
            Write-Host "  isolated datasets from Kaggle Native API, training cross-dependent computational"
            Write-Host "  topologies locally, and then aggressively PURGING them from your Windows SSD"
            Write-Host "  immediately to preserve hard drive bounds while maximizing local GPUs."
            Write-Host ""
            if (Test-Environment) {
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" "smart_orchestrator.py"
                Pop-Location
            }
            Read-Host "`nPress Enter to return to menu..."
        }
        '6' {
            Write-Header "SINGLE-EPOCH UNIT TEST (ALL MODELS)"
            Write-Host "  [DESCRIPTION]" -ForegroundColor White
            Write-Host "  This initiates a fully rigorous automated structural diagnostic test,"
            Write-Host "  forcing all 21 models natively to train precisely 1 single epoch."
            Write-Host "  Perfect for completely validating memory buffers cleanly."
            Write-Host ""
            if (Test-Environment) {
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" "train_all.py" --epochs 1
                Pop-Location
            }
            Read-Host "`nPress Enter to return to menu..."
        }
        '7' {
            Write-Host "`nGoodbye!" -ForegroundColor Yellow
            return
        }
        default {
            Write-Host "`nInvalid selection. Please try again." -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
}
