# training/lemgendary_hub.ps1 [Refresh: 2026-03-27_23:50]
# Master Orchestration Script for LemGendary AI Training & Management

$script:HUB_DIR = $PSScriptRoot
if (-not $script:HUB_DIR) { $script:HUB_DIR = Get-Location }

# PowerShell 5.1 compatibility for Join-Path
$script:VENV_DIR = Join-Path $script:HUB_DIR ".venv"
$script:REQ_FILE = Join-Path $script:HUB_DIR "requirements.txt"

function Unlock-Environment {
    Write-Host "  [*] Checking for active environment locks..." -ForegroundColor Gray
    $lockedProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*$script:VENV_DIR*" }
    if ($lockedProcs) {
        Write-Host "  [!] WARNING: Active Python processes are locking the .venv!" -ForegroundColor Yellow
        Write-Host "  Please close all other terminals or training runs using this environment." -ForegroundColor Red
        $lockedProcs | ForEach-Object { Write-Host "      -> PID: $($_.Id) | Path: $($_.Path)" -ForegroundColor Gray }
        $choice = Read-Host "  Would you like me to attempt a FORCE KILL (Nuke) to release locks? (Y/N)"
        if ($choice -eq 'Y' -or $choice -eq 'y') {
            Clear-EnvironmentLocks
        } else {
            Read-Host "  Press Enter once you have closed the conflicting apps manually to continue..."
        }
    }
}

function Clear-EnvironmentLocks {
    Write-Host "  [!] EXECUTING INDESTRUCTIBLE LOCK CLEARANCE..." -ForegroundColor Magenta
    $lockedProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*$script:VENV_DIR*" }
    foreach ($proc in $lockedProcs) {
        try {
            Write-Host "      -> Terminanting locked PID: $($proc.Id)..." -ForegroundColor Gray
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } catch {
            Write-Host "      [!] Failed to terminate PID: $($proc.Id). Elevate to Admin if persistence remains." -ForegroundColor Yellow
        }
    }
    Start-Sleep -Seconds 1
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
            Initialize-Environment
            return $true
        }
        return $false
    }

    # Pre-Flight Audit
    Write-Header "ENVIRONMENT INTEGRITY AUDIT"
    Write-Host "  [*] Verifying core library specialization (PyYAML / Torch / DirectML)..." -ForegroundColor Gray
    $auditCmd = "import yaml; print('YAML_PATH: ' + yaml.__file__); import torch; print('Torch: ' + torch.__version__); print('CUDA Ready: ' + str(torch.cuda.is_available())); print('Device: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
    $auditResult = & "$script:VENV_DIR\Scripts\python.exe" -c $auditCmd 2>&1
    
    $venvBase = Split-Path $script:VENV_DIR -Leaf
    if ($auditResult -match "YAML_PATH: .*$venvBase" -and $auditResult -match "Torch:") {
        Write-Host "  [PASS] Integrity Audit Successful. Environment is healthy." -ForegroundColor Green
        if ($auditResult -match "CUDA Ready: True") {
            Write-Host "  [ACCELERATED] NVIDIA Hardware detected and linked." -ForegroundColor Cyan
        } else {
            Write-Host "  [WARNING] Running in CPU mode. Check NVIDIA drivers." -ForegroundColor Yellow
        }
        return $true
    } else {
        Write-Host "  [FAIL] Integrity Audit Failed! Core libraries missing or corrupted." -ForegroundColor Red
        Write-Host "  Suggested Fix: Run Option 1 again." -ForegroundColor White
        return $false
    }
}

function Initialize-Environment {
    Write-Header "PREPARING PYTHON 3.12 ENVIRONMENT"
    $targetPython = "3.12"
    
    # 1. Advanced Discovery: Find SYSTEM Python (Skip .venv paths)
    $pyPath = Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike "*\.venv\*" } | Select-Object -First 1 -ExpandProperty Source
    $knownSystemPath = "C:\Users\lemtr\AppData\Local\Programs\Python\Python312\python.exe"

    if ($null -eq $pyPath -or (& $pyPath --version) -notmatch $targetPython) {
        if (Test-Path $knownSystemPath) {
            $pyPath = $knownSystemPath
            Write-Host "  [+] Recovered System Python at $pyPath" -ForegroundColor Cyan
        } else {
            $searchPath = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
            if (Test-Path $searchPath) {
                $pyPath = $searchPath
            } else {
                Write-Host "  [!] Python 3.12 not found. Attempting winget install..." -ForegroundColor Yellow
                try {
                    winget install --id "Python.Python.3.12" -e --scope user --silent --accept-package-agreements --accept-source-agreements
                    if (Test-Path $searchPath) { $pyPath = $searchPath }
                } catch {
                    Write-Host "  [ERROR] Auto-install failed. Please install manually." -ForegroundColor Red
                    return
                }
            }
        }
    }
    
    if (-not (Test-Path $pyPath)) {
        Write-Host "  [ERROR] Fatal: Could not identify a valid System Python binary." -ForegroundColor Red
        return
    }

    # Construction Reset
    if (Test-Path $script:VENV_DIR) {
        Write-Host "  [!] NUKING stale environment to ensure structural integrity..." -ForegroundColor Magenta
        $retryCount = 0
        while ($retryCount -lt 5) {
            try { Remove-Item -Path $script:VENV_DIR -Recurse -Force -ErrorAction Stop; break }
            catch { $retryCount++; Clear-EnvironmentLocks; Start-Sleep -Seconds 1 }
        }
    }

    Write-Host "  [1/4] Constructing virtual environment (using System Python)..." -ForegroundColor Cyan
    & $pyPath -m venv $script:VENV_DIR
    $venvPy = "$script:VENV_DIR\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { return }

    Write-Host "  [2/4] Initializing environment (pip upgrade)..." -ForegroundColor Cyan
    & $venvPy -m pip install --upgrade pip
    
    Write-Host "  [3/4] Synchronizing AI Core (PyTorch + Hardware Backends)..." -ForegroundColor Cyan
    # Detect NVIDIA vs AMD for optimized index selection
    $isNvidia = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" })
    
    if ($null -ne $isNvidia) {
        Write-Host "  [+] NVIDIA GPU Detected: GTX/RTX hardware optimization active." -ForegroundColor Green
        & $venvPy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
        & $venvPy -m pip install onnxruntime-gpu --force-reinstall
    } else {
        & $venvPy -m pip install --upgrade torchruntime
        & $venvPy -m torchruntime install --auto
        if ((& $venvPy -c "import torch; print(torch.cuda.is_available())") -eq "False") {
            & $venvPy -m pip install onnxruntime-directml
        }
    }

    Write-Host "  [4/4] Installing Auxiliary libraries (PyYAML/Datasets/ONNX)..." -ForegroundColor Cyan
    & $venvPy -m pip install --force-reinstall pyyaml
    & $venvPy -m pip install -r $script:REQ_FILE

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
        '1' { Initialize-Environment; Read-Host "Press Enter to return..." }
        '2' {
            if (Test-Environment) {
                $env:PYTHONPATH=""; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "training/train.py"; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '3' {
            if (Test-Environment) {
                $env:PYTHONPATH=""; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "train_all.py"; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '4' {
            Write-Header "DEPLOY TO KAGGLE CLOUD"
            Write-Host "  Open Kaggle -> Create Notebook -> File -> Import Notebook."
            Read-Host "Press Enter to return..."
        }
        '5' {
            if (Test-Environment) {
                $env:PYTHONPATH=""; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "smart_orchestrator.py"; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '6' {
            if (Test-Environment) {
                $env:PYTHONPATH=""; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "train_all.py" --epochs 1 --yes; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '7' { return }
        default { Write-Host "Invalid selection."; Start-Sleep -Seconds 1 }
    }
}
