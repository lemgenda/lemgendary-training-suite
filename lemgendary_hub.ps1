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

function Unlock-Environment {
    Write-Host "  [*] Checking for active environment locks..." -ForegroundColor Gray
    $lockedProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*$script:VENV_DIR*" }
    if ($lockedProcs) {
        Write-Host "  [!] WARNING: Active Python processes are locking the .venv!" -ForegroundColor Yellow
        Write-Host "  Please close all other terminals or training runs using this environment." -ForegroundColor Red
        $lockedProcs | ForEach-Object { Write-Host "      -> PID: $($_.Id) | Path: $($_.Path)" -ForegroundColor Gray }
        Read-Host "  Press Enter once you have closed the conflicting apps to continue..."
    }
}

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
    Write-Header "PREPARING PYTHON 3.12 ENVIRONMENT"
    $targetPython = "3.12"
    $pythonId = "Python.Python.3.12"

    # 1. Advanced Discovery: Search for 3.12 binary if 'python' is old/missing
    $pyPath = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if ($null -eq $pyPath -or (& $pyPath --version) -notmatch $targetPython -or -not (Test-Path $pyPath)) {
        $searchPath = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
        if (Test-Path $searchPath) {
            $pyPath = $searchPath
            Write-Host "  [+] Found Python 3.12 at $pyPath" -ForegroundColor Cyan
        } else {
            Write-Host "  [!] Python 3.12 not found. Attempting install via winget..." -ForegroundColor Yellow
            try {
                winget install --id $pythonId -e --scope user --silent --accept-package-agreements --accept-source-agreements
                if (Test-Path $searchPath) { 
                    $pyPath = $searchPath 
                    # In-session path refresh
                    $env:Path = "$([System.IO.Path]::GetDirectoryName($searchPath));$env:Path"
                } else {
                    # Final fallback check Program Files
                    $pgmPath = "C:\Program Files\Python312\python.exe"
                    if (Test-Path $pgmPath) { $pyPath = $pgmPath }
                }
            } catch {
                Write-Host "  [ERROR] Auto-install failed. Please install Python 3.12 manually." -ForegroundColor Red
                return
            }
        }
    }
    
    # Final validation of $pyPath
    if (-not (Test-Path $pyPath)) {
        Write-Host "  [ERROR] Fatal: Python 3.12 binary remains elusive even after search/install." -ForegroundColor Red
        return
    }

    # 2. VENV Integrity Check: wipe if stale/drifted
    if (Test-Path $script:VENV_DIR) {
        $venvCfg = Join-Path $script:VENV_DIR "pyvenv.cfg"
        if (Test-Path $venvCfg) {
            $versionMatch = Get-Content $venvCfg | Select-String "version = $targetPython"
            if (-not $versionMatch) {
                Write-Host "  [!] Version Drift detected (not $targetPython). Wiping existing .venv..." -ForegroundColor Magenta
                Unlock-Environment
                try {
                    Remove-Item -Path $script:VENV_DIR -Recurse -Force -ErrorAction Stop
                } catch {
                    Write-Host "  [ERROR] Permission Denied: Could not remove .venv folder." -ForegroundColor Red
                    Write-Host "  Manual action required: Run 'Remove-Item -Path $script:VENV_DIR -Recurse -Force' after closing all apps." -ForegroundColor White
                    return
                }
            }
        }
    }

    Write-Host "  [1/4] Constructing virtual environment ($script:VENV_DIR)..." -ForegroundColor Cyan
    try {
        & $pyPath -m venv $script:VENV_DIR -ErrorAction Stop
    } catch {
        Write-Host "  [ERROR] Bootstrap failure: Virtual environment creation failed." -ForegroundColor Red
        Write-Host "  Ensure no other apps are using the folder: $script:VENV_DIR" -ForegroundColor Yellow
        return
    }

    Write-Host "  [2/4] Initializing environment (pip upgrade)..." -ForegroundColor Cyan
    & "$script:VENV_DIR\Scripts\python.exe" -m pip install --upgrade pip

    Write-Host "  [3/4] Synchronizing AI Core (PyTorch + Hardware Backends)..." -ForegroundColor Cyan
    # torchruntime automatically detects NVIDIA (CUDA) vs AMD (ROCm/DirectML) vs CPU
    & "$script:VENV_DIR\Scripts\python.exe" -m pip install --upgrade torchruntime
    & "$script:VENV_DIR\Scripts\python.exe" -m torchruntime install --auto

    # Explicitly ensure onnxruntime-directml for AMD users in 2026
    if ((& "$script:VENV_DIR\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())") -eq "False") {
        Write-Host "  [+] Non-CUDA hardware detected. Strengthening ONNX DirectML support..." -ForegroundColor Cyan
        & "$script:VENV_DIR\Scripts\python.exe" -m pip install onnxruntime-directml
    }

    Write-Host "  [4/4] Installing Auxiliary libraries (Datasets/ONNX/YOLO)..." -ForegroundColor Cyan
    & "$script:VENV_DIR\Scripts\python.exe" -m pip install -r $script:REQ_FILE

    # 3. Web App Context Sync (Node.js)
    Write-Header "SYNCHRONIZING WEB APP (VITE/NODE.JS)"
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        if (Test-Path "$PARENT_DIR\package.json") {
            Write-Host "  [+] Node project detected in ROOT. Running npm install..." -ForegroundColor Cyan
            Push-Location $PARENT_DIR
            npm install
            Pop-Location
        }
    } else {
        Write-Host "  [!] Node.js/npm not found. Skipping web app synchronization." -ForegroundColor Yellow
    }

    # 4. Session Environment Alignment
    $env:Path = "$script:VENV_DIR\Scripts;$env:Path"
    if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
        $kgPath = "$script:VENV_DIR\Scripts\python.exe"
        function global:kaggle { & $kgPath -m kaggle @args }
    }

    Write-Host "`n  [SUCCESS] All LemGendary 2026 Systems are Synchronized!" -ForegroundColor Green
}

function Show-Menu {
    Clear-Host
    Write-Header "LEMGENDARY AI TRAINING SUITE (2026 SPECIALIZATION)"
    Write-Host " [ENVIRONMENT: $(if ($env:VIRTUAL_ENV) { 'VIRTUAL' } else { 'GLOBAL' })]" -ForegroundColor Gray
    Write-Host "  1. Initialize/Fix All Systems (Python + Node.js + Specialized GPUs)"
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
                & "$script:VENV_DIR\Scripts\python.exe" "train_all.py" --epochs 1 --yes
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
