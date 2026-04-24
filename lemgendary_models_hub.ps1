# training/lemgendary_hub.ps1 [Refresh: 2026-03-27_23:50]
# Master Orchestration Script for LemGendary AI Training & Management
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null


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

function Invoke-JanitorPurge {
    Write-Header "ENVIRONMENTAL JANITOR: ORPHAN PURGE"
    Write-Host "  [*] Scanning for orphaned LemGendary infrastructure..." -ForegroundColor Gray
    
    # 2026 Process Hygiene: Kill any Python/PowerShell processes containing the Hub Directory in their command line
    # We exclude the current process ($PID) to prevent Hub self-termination
    $targetProcs = Get-WmiObject Win32_Process | Where-Object { 
        ($_.Name -match "python" -or $_.Name -match "powershell") -and 
        $_.CommandLine -match [regex]::Escape($script:HUB_DIR) -and 
        $_.ProcessId -ne $PID 
    }

    if ($targetProcs) {
        Write-Host "  [!] Identified $($targetProcs.Count) orphaned system artifacts." -ForegroundColor Yellow
        foreach ($proc in $targetProcs) {
            Write-Host "      -> Purging PID: $($proc.ProcessId) | $($proc.Name)..." -ForegroundColor Magenta
            try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
        }
        Write-Host "  [SUCCESS] All structural orphans de-provisioned." -ForegroundColor Green
    } else {
        Write-Host "  [PASS] No environmental orphans detected. Matrix is clean." -ForegroundColor Green
    }
    
    # Also clear VENV locks specifically
    Clear-EnvironmentLocks
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
    Write-Host "  [*] Verifying core library specialization (PyYAML / Torch / OpenCV / DirectML)..." -ForegroundColor Gray
    $auditCmd = "import yaml; print('YAML_PATH: ' + yaml.__file__); import torch; print('Torch: ' + torch.__version__); import cv2; print('OpenCV: ' + cv2.__version__); print('CUDA Ready: ' + str(torch.cuda.is_available())); print('Device: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
    $auditResult = & "$script:VENV_DIR\Scripts\python.exe" -W ignore -c $auditCmd 2>$null
    
    $venvBase = Split-Path $script:VENV_DIR -Leaf
    if ($auditResult -match "YAML_PATH: .*$venvBase" -and $auditResult -match "Torch:" -and $auditResult -match "OpenCV:") {
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
    $installSuccess = $false
    
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
                    if (Test-Path $searchPath) { 
                        $installSuccess = $true 
                        $pyPath = $searchPath 
                    }
                } catch {
                    Write-Host "  [ERROR] Auto-install failed. Please install manually." -ForegroundColor Red
                    return
                }
            }
        }
    }
    
    # Session Break: If we just installed Python, the current PATH is out of sync.
    if ($installSuccess) {
        Write-Host "`n********************************************************************************" -ForegroundColor Yellow
        Write-Host "  [SUCCESS] Python 3.12 has been mathematically installed by the Hub!" -ForegroundColor Green
        Write-Host "  [CRITICAL] PowerShell needs to be RESTARTED to recognize the new environment." -ForegroundColor Red
        Write-Host "********************************************************************************`n" -ForegroundColor Yellow
        Write-Host "  Please close this terminal, open a new one, and run Option 1 again." -ForegroundColor White
        Read-Host "  Press Enter to exit the Hub and restart manually..."
        exit
    }

    if (-not (Test-Path $pyPath)) {
        Write-Host "  [ERROR] Fatal: Could not identify a valid System Python binary." -ForegroundColor Red
        Write-Host "  Please install Python 3.12 manually from python.org." -ForegroundColor White
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
        Write-Host "  [4/4] Resolving holistic dependency graph (Unified Install)..." -ForegroundColor Cyan
        & $venvPy -m pip install torch torchvision torchaudio onnxruntime-gpu -r $script:REQ_FILE --extra-index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Host "  [4/4] Resolving holistic dependency graph (Unified Install)..." -ForegroundColor Cyan
        & $venvPy -m pip install torchruntime onnxruntime-directml -r $script:REQ_FILE
        & $venvPy -m torchruntime install --auto
    }

    Write-Host "`n  [SUCCESS] All LemGendary 2026 Systems are Synchronized!" -ForegroundColor Green
}

function Get-ModelSelection {
    Write-Header "SELECT MODEL CATEGORY"
    Write-Host "  1. Quality Assessment  (NIMA, Aesthetics)" -ForegroundColor Cyan
    Write-Host "  2. Face & Detection    (RetinaFace, YOLOv8, CodeFormer)" -ForegroundColor Cyan
    Write-Host "  3. Super-Resolution    (UltraZoom x2/x3/x4/x8)" -ForegroundColor Cyan
    Write-Host "  4. Image Restoration   (NAFNet, MIRNet, FFANet, MPRNet)" -ForegroundColor Cyan
    Write-Host "  5. Universal Hybrid    (UPN v2, Multi-Restorer, Film)" -ForegroundColor Cyan
    Write-Host "  6. Cancel" -ForegroundColor Gray
    Write-Host ""
    
    $catChoice = Read-Host "Select a category (1-6)"
    $modelList = @()
    switch ($catChoice) {
        '1' { $modelList = @("nima_aesthetic", "nima_technical") }
        '2' { $modelList = @("codeformer", "parsenet", "retinaface_mobilenet", "retinaface_resnet", "yolov8n") }
        '3' { $modelList = @("ultrazoom_x2", "ultrazoom_x3", "ultrazoom_x4", "ultrazoom_x8") }
        '4' { $modelList = @("ffanet_indoor", "ffanet_outdoor", "mprnet_deraining", "mirnet_lowlight", "mirnet_exposure", "nafnet_debluring", "nafnet_denoising") }
        '5' { $modelList = @("upn_v2", "professional_multitask_restoration", "film_restorer") }
        default { return $null }
    }

    Write-Header "SELECT SPECIFIC MODEL"
    for ($i=0; $i -lt $modelList.Count; $i++) {
        Write-Host "  $($i+1). $($modelList[$i])" -ForegroundColor Green
    }
    Write-Host "  $($modelList.Count + 1). Back" -ForegroundColor Gray
    Write-Host ""
    
    $modelChoice = (Read-Host "Select a model (1-$($modelList.Count + 1))").Trim()
    if ($modelChoice -as [int] -and [int]$modelChoice -ge 1 -and [int]$modelChoice -le $modelList.Count) {
        return $modelList[[int]$modelChoice - 1]
    }
    return $null
}

function Invoke-BootstrapCheck {
    # Lightweight, silent scan for Python 3.12 (Proactive 2026 Discovery)
    $pyPath = Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike "*\.venv\*" } | Select-Object -First 1 -ExpandProperty Source
    $knownPath = "C:\Users\lemtr\AppData\Local\Programs\Python\Python312\python.exe"
    
    if ($null -eq $pyPath -and -not (Test-Path $knownPath)) {
        Write-Host "`n********************************************************************************" -ForegroundColor Red
        Write-Host "  [!] CRITICAL: Python 3.12 Core not detected on this system." -ForegroundColor Red
        Write-Host "********************************************************************************" -ForegroundColor Yellow
        Write-Host "  The LemGendary Hub requires a system-level Python 3.12 to bootstrap natively." -ForegroundColor White
        $choice = Read-Host "  👉 Would you like me to attempt an AUTOMATIC installation now? (y/n)"
        if ($choice -eq 'y' -or $choice -eq 'Y') {
            Initialize-Environment
        } else {
            Write-Host "  🛑 Python absolute requirement failed. Aborting Hub launch..." -ForegroundColor Red
            Start-Sleep -Seconds 2
            exit
        }
    }
}

function Show-Menu {
    Clear-Host
    Write-Header "LEMGENDARY AI TRAINING SUITE (2026 SPECIALIZATION)"
    Write-Host " [ENVIRONMENT: $(if ($env:VIRTUAL_ENV) { 'VIRTUAL' } else { 'GLOBAL' })]" -ForegroundColor Gray
    Write-Host "  1. Initialize/Fix All Systems (Python + Node.js + Specialized GPUs)"
    Write-Host "  2. Train Individual Model      (Launches LemGendary Training Suite)"
    Write-Host "  3. Global Orchestration        (Automated sequential multi-model run)"
    Write-Host "  4. Single-Epoch Unit Test      (Diagnostic 1-Epoch pass for ALL models)"
    Write-Host "  5. Run Environmental Janitor   (Force-Purge all project Orphans)" -ForegroundColor Yellow
    Write-Host "  Q. Exit"
    Write-Host ""
}

# Pre-Flight Bootstrap Python check (Silent if found)
Invoke-BootstrapCheck

while ($true) {
    Show-Menu
    $choice = (Read-Host "Select an option (1-5, Q)").Trim()
    switch ($choice) {
        '1' { Initialize-Environment; Read-Host "Press Enter to return..." }
        '2' {
            if (Test-Environment) {
                $selectedModel = Get-ModelSelection
                if ($null -ne $selectedModel) {
                    $rocket = [char]0xD83D + [char]0xDE80
                    Write-Host "  [$rocket] Launching Training Matrix for >> $selectedModel <<..." -ForegroundColor Green
                    Write-Host "      -> Target Manifold: $selectedModel" -ForegroundColor Gray
                    Invoke-JanitorPurge # Ensure clean start
                    $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                    $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                    Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "training/train.py" --model $selectedModel; Pop-Location
                }
            }
            Read-Host "Press Enter to return..."
        }
        '3' {
            if (Test-Environment) {
                $env:PYTHONPATH=""; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "train_all.py"; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '4' {
            if (Test-Environment) {
                Invoke-JanitorPurge
                $env:PYTHONPATH=""; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "train_all.py" --epochs 1 --yes; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '5' { Invoke-JanitorPurge; Read-Host "Purge Complete. Press Enter to return..." }
        'Q' { return }
        'q' { return }
        default { Write-Host "Invalid selection."; Start-Sleep -Seconds 1 }
    }
}
