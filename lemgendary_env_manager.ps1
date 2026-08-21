# lemgendary_env_manager.ps1 - Provisioning & Environment Validation
$script:HUB_DIR = $PSScriptRoot
if (-not $script:HUB_DIR) { $script:HUB_DIR = Get-Location }
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
    Write-Host "  [*] Verifying core library specialization (Torch / PEFT / Diffusers / BitsAndBytes)..." -ForegroundColor Gray
    $auditCmd = "import yaml; import torch; import diffusers; import peft; import bitsandbytes; print('CUDA Ready: ' + str(torch.cuda.is_available())); print('Device: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
    $auditResult = & "$script:VENV_DIR\Scripts\python.exe" -W ignore -c $auditCmd 2>$null
    
    if ($auditResult -match "CUDA Ready:") {
        Write-Host "  [PASS] Integrity Audit Successful. SOTA libraries verified." -ForegroundColor Green
        if ($auditResult -match "CUDA Ready: True") {
            Write-Host "  [ACCELERATED] NVIDIA Hardware detected and linked." -ForegroundColor Cyan
        } else {
            Write-Host "  [WARNING] Running in CPU mode. Check NVIDIA drivers." -ForegroundColor Yellow
        }
        return $true
    } else {
        Write-Host "  [FAIL] Integrity Audit Failed! SOTA libraries (PEFT/Diffusers/BNB) missing." -ForegroundColor Red
        Write-Host "  Suggested Fix: Run Option 1 to repair dependencies." -ForegroundColor White
        return $false
    }
}

function Initialize-Environment {
    Write-Header "INITIALIZING / FIXING SYSTEMS (PYTHON + NODE.JS + GPU)"
    $targetPython = "3.12"
    $installSuccess = $false
    
    Write-Host "  [*] Checking for Node.js..." -ForegroundColor Gray
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Write-Host "  [!] Node.js not found. Attempting winget install..." -ForegroundColor Yellow
        try {
            winget install --id "OpenJS.NodeJS" -e --scope user --silent --accept-package-agreements --accept-source-agreements
            Write-Host "  [+] Node.js installed." -ForegroundColor Green
            $installSuccess = $true
        } catch {
            Write-Host "  [ERROR] Auto-install for Node.js failed. Please install manually." -ForegroundColor Red
        }
    } else {
        Write-Host "  [PASS] Node.js is installed." -ForegroundColor Green
    }

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
    
    # Session Break: If we just installed Python or Node, the current PATH is out of sync.
    if ($installSuccess) {
        Write-Host "`n********************************************************************************" -ForegroundColor Yellow
        Write-Host "  [SUCCESS] Missing core systems (Python/Node) have been mathematically installed!" -ForegroundColor Green
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

    if (-not (Test-Path $script:VENV_DIR)) {
        Write-Host "  [1/4] Constructing virtual environment (using System Python)..." -ForegroundColor Cyan
        & $pyPath -m venv $script:VENV_DIR
    } else {
        Write-Host "  [1/4] Virtual environment exists. Skipping construction..." -ForegroundColor Cyan
    }
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
        & $venvPy -m pip install -r $script:REQ_FILE --extra-index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Host "  [4/4] Resolving holistic dependency graph (Unified Install)..." -ForegroundColor Cyan
        & $venvPy -m pip install torchruntime onnxruntime-directml -r $script:REQ_FILE
        & $venvPy -m torchruntime install --auto
    }

    Write-Host "`n  [SUCCESS] All LemGendary 2026 Systems are Synchronized!" -ForegroundColor Green
}

function Get-ModelSelection {
    while ($true) {
        Write-Header "SELECT MODEL DOMAIN (PARENT CATEGORY)"
        Write-Host "  1. Image Manipulation & Restoration" -ForegroundColor Cyan
        Write-Host "  2. Image Generation & Multimodal" -ForegroundColor Cyan
        Write-Host "  3. Financial & Time-Series" -ForegroundColor Cyan
        Write-Host "  Q. Cancel / Return" -ForegroundColor Gray
        Write-Host ""

        $domainChoice = (Read-Host "Select a domain (1-3, Q)").Trim()
        if ($domainChoice -eq 'Q' -or $domainChoice -eq 'q') { return $null }

        switch ($domainChoice) {
            '1' {
                while ($true) {
                    Write-Header "IMAGE MANIPULATION & RESTORATION - SUB-CATEGORIES"
                    Write-Host "  1. Quality Assessment  (NIMA Aesthetics, Technical, Authenticity, Universal NSFW)" -ForegroundColor Cyan
                    Write-Host "  2. Face & Detection    (RetinaFace, YOLOv8, CodeFormer, ParseNet)" -ForegroundColor Cyan
                    Write-Host "  3. Super-Resolution    (UltraZoom x2/x3/x4/x8)" -ForegroundColor Cyan
                    Write-Host "  4. Image Restoration   (NAFNet, MIRNet, FFANet, MPRNet)" -ForegroundColor Cyan
                    Write-Host "  5. Universal Hybrid    (UPN v2, Multi-Restorer, Film Restoration)" -ForegroundColor Cyan
                    Write-Host "  B. Back to Domain Menu" -ForegroundColor Gray
                    Write-Host ""

                    $subChoice = (Read-Host "Select sub-category (1-5, B)").Trim()
                    if ($subChoice -eq 'B' -or $subChoice -eq 'b') { break }

                    $modelList = @()
                    switch ($subChoice) {
                        '1' { $modelList = @("nima_aesthetic_mobile", "nima_aesthetic_efficientnet", "nima_aesthetic_pro", "nima_technical", "nima_authenticity", "universal_nsfw_classification") }
                        '2' { $modelList = @("codeformer", "parsenet", "retinaface_mobilenet", "yolov8n") }
                        '3' { $modelList = @("ultrazoom") }
                        '4' { $modelList = @("ffanet_indoor", "ffanet_outdoor", "mprnet_deraining", "mirnet_lowlight", "mirnet_exposure", "nafnet_debluring", "nafnet_denoising") }
                        '5' { $modelList = @("upn_v2", "professional_multitask_restoration", "film_restorer") }
                        default { continue }
                    }

                    $selected = Show-ModelList -Models $modelList
                    if ($null -ne $selected) { return $selected }
                }
            }

            '2' {
                while ($true) {
                    Write-Header "IMAGE GENERATION & MULTIMODAL - SUB-CATEGORIES"
                    Write-Host "  1. Generative Diffusion (SDXL, Flux)" -ForegroundColor Cyan
                    Write-Host "  2. Vision-Language      (LLaVA, BLIP-2)" -ForegroundColor Cyan
                    Write-Host "  B. Back to Domain Menu" -ForegroundColor Gray
                    Write-Host ""

                    $subChoice = (Read-Host "Select sub-category (1-2, B)").Trim()
                    if ($subChoice -eq 'B' -or $subChoice -eq 'b') { break }

                    $modelList = @()
                    switch ($subChoice) {
                        '1' { $modelList = @("diffusion_sdxl", "diffusion_flux") }
                        '2' { $modelList = @("vlm_llava", "vlm_blip2") }
                        default { continue }
                    }

                    $selected = Show-ModelList -Models $modelList
                    if ($null -ne $selected) { return $selected }
                }
            }

            '3' {
                # --- DOMAIN 3: Financial & Time-Series ---
                while ($true) {
                    Write-Header "FINANCIAL & TIME-SERIES - SUB-CATEGORIES"
                    Write-Host "  1. Forex Trading (ForexPredictor Multi-Scale CNN-Transformer)" -ForegroundColor Cyan
                    Write-Host "  B. Back to Domain Menu" -ForegroundColor Gray
                    Write-Host ""

                    $subChoice = (Read-Host "Select sub-category (1, B)").Trim()
                    if ($subChoice -eq 'B' -or $subChoice -eq 'b') { break }

                    $modelList = @()
                    switch ($subChoice) {
                        '1' { $modelList = @("forex_predictor") }
                        default { continue }
                    }

                    $selected = Show-ModelList -Models $modelList
                    if ($null -ne $selected) { return $selected }
                }
            }

            default {
                Write-Host "Invalid selection." -ForegroundColor Red
                Start-Sleep -Seconds 1
            }
        }
    }
}

function Show-ModelList {
    param([string[]]$Models)
    if ($Models.Count -eq 1) { return $Models[0] }
    Write-Header "SELECT SPECIFIC MODEL MANIFOLD"
    for ($i=0; $i -lt $Models.Count; $i++) {
        Write-Host "  $($i+1). $($Models[$i])" -ForegroundColor Green
    }
    Write-Host "  $($Models.Count + 1). Back" -ForegroundColor Gray
    Write-Host ""

    $choice = (Read-Host "Select a model (1-$($Models.Count + 1))").Trim()
    if ($choice -as [int] -and [int]$choice -ge 1 -and [int]$choice -le $Models.Count) {
        return $Models[[int]$choice - 1]
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
        $choice = Read-Host "  [?] Would you like me to attempt an AUTOMATIC installation now? (y/n)"
        if ($choice -eq 'y' -or $choice -eq 'Y') {
            Initialize-Environment
        } else {
            Write-Host "  [ERROR] Python absolute requirement failed. Aborting Hub launch..." -ForegroundColor Red
            Start-Sleep -Seconds 2
            exit
        }
    }
}

