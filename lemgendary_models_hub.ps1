# training/lemgendary_hub.ps1 [Refresh: 2026-08-18_11:15]
# Master Orchestration Script for LemGendary AI Training & Management
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null


$script:HUB_DIR = $PSScriptRoot
if (-not $script:HUB_DIR) { $script:HUB_DIR = Get-Location }

$script:VENV_DIR = Join-Path $script:HUB_DIR ".venv"
$script:REQ_FILE = Join-Path $script:HUB_DIR "requirements.txt"
$script:ENV_MGR = Join-Path (Split-Path -Parent $script:HUB_DIR) "lemgendary-env-manager\lemgendary_env_manager.ps1"

function Write-Ui {
    # Show a line to the user on the information stream (visible by default,
    # capturable with 6>), coloured with ANSI codes when a colour is named.
    param([string]$Message = "", [string]$Color = "")
    $codes = @{
        Black = 30; DarkRed = 31; DarkGreen = 32; DarkYellow = 33; DarkBlue = 34; DarkMagenta = 35
        DarkCyan = 36; Gray = 37; DarkGray = 90; Red = 91; Green = 92; Yellow = 93; Blue = 94
        Magenta = 95; Cyan = 96; White = 97
    }
    if ($Color -and $codes.ContainsKey($Color)) {
        $esc = [char]27
        $Message = "$esc[$($codes[$Color])m$Message$esc[0m"
    }
    Write-Information -MessageData $Message -InformationAction Continue
}

function Write-Header($text) {
    Write-Ui "`n================================================================================" -Color Cyan
    Write-Ui "  $text" -Color White
    Write-Ui "================================================================================`n" -Color Cyan
}

function Invoke-BootstrapCheck {
    # Handled by LemGendary Environment Manager
}

function Initialize-Environment {
    Write-Header "INITIALIZING ENVIRONMENT"
    if (Test-Path $script:ENV_MGR) {
        Write-Ui "  [*] Delegating to LemGendary Environment Manager..." -Color Cyan
        & $script:ENV_MGR install --project lemgendary-training-suite
    } else {
        Write-Ui "  [!] LemGendary Environment Manager not found at $script:ENV_MGR. Falling back to local python -m venv." -Color Yellow
        try {
            $GlobalPy = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
        } catch [System.Management.Automation.CommandNotFoundException] {
            Write-Ui "  [!] No global python found on PATH; cannot create $script:VENV_DIR." -Color Red
            return
        }
        & $GlobalPy -m venv $script:VENV_DIR
        & "$script:VENV_DIR\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools
        if (Test-Path $script:REQ_FILE) {
            & "$script:VENV_DIR\Scripts\python.exe" -m pip install -r $script:REQ_FILE
        }
    }
}

function Test-Environment {
    $py = "$script:VENV_DIR\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        Write-Ui "  [!] Virtual environment not detected at $script:VENV_DIR" -Color Yellow
        $choice = Read-Host "  Would you like to initialize it now? (Y/N)"
        if ($choice -eq 'Y' -or $choice -eq 'y') {
            Initialize-Environment
            return (Test-Path $py)
        }
        return $false
    }
    return $true
}

function Invoke-JanitorPurge {
    $lockedProcs = Get-Process | Where-Object { $_.ProcessName -eq 'python' -and $_.Path -like "*$script:VENV_DIR*" }
    foreach ($proc in $lockedProcs) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } catch {
            Write-Warning "Could not stop python process $($proc.Id): $($_.Exception.Message)"
        }
    }
}

function Get-ModelSelection {
    while ($true) {
        Write-Header "SELECT MODEL DOMAIN (PARENT CATEGORY)"
        Write-Ui "  1. Image Manipulation & Restoration" -Color Cyan
        Write-Ui "  2. Image Generation & Multimodal" -Color Cyan
        Write-Ui "  3. Financial & Time-Series" -Color Cyan
        Write-Ui "  Q. Cancel / Return" -Color Gray
        Write-Ui ""

        $domainChoice = (Read-Host "Select a domain (1-3, Q)").Trim()
        if ($domainChoice -eq 'Q' -or $domainChoice -eq 'q') { return $null }

        switch ($domainChoice) {
            '1' {
                # --- DOMAIN 1: Image Manipulation & Restoration ---
                while ($true) {
                    Write-Header "IMAGE MANIPULATION & RESTORATION - SUB-CATEGORIES"
                    Write-Ui "  1. Quality Assessment  (NIMA Aesthetics, Technical, Authenticity, Universal NSFW)" -Color Cyan
                    Write-Ui "  2. Face & Detection    (RetinaFace, YOLOv8, CodeFormer, ParseNet)" -Color Cyan
                    Write-Ui "  3. Super-Resolution    (UltraZoom x2/x3/x4/x8)" -Color Cyan
                    Write-Ui "  4. Image Restoration   (NAFNet, MIRNet, FFANet, MPRNet)" -Color Cyan
                    Write-Ui "  5. Universal Hybrid    (UPN v2, Multi-Restorer, Film Restoration)" -Color Cyan
                    Write-Ui "  B. Back to Domain Menu" -Color Gray
                    Write-Ui ""

                    $subChoice = (Read-Host "Select sub-category (1-5, B)").Trim()
                    if ($subChoice -eq 'B' -or $subChoice -eq 'b') { break }

                    $modelList = @()
                    switch ($subChoice) {
                        '1' { $modelList = @("nima_aesthetic_mobile", "nima_aesthetic_efficientnet", "nima_aesthetic_pro", "nima_technical", "nima_authenticity", "universal_nsfw_classification") }
                        '2' { $modelList = @("codeformer", "parsenet", "retinaface", "yolov8n") }
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
                # --- DOMAIN 2: Image Generation & Multimodal ---
                while ($true) {
                    Write-Header "IMAGE GENERATION & MULTIMODAL - SUB-CATEGORIES"
                    Write-Ui "  1. Generative Diffusion (SDXL, Flux)" -Color Cyan
                    Write-Ui "  2. Vision-Language      (LLaVA, BLIP-2)" -Color Cyan
                    Write-Ui "  B. Back to Domain Menu" -Color Gray
                    Write-Ui ""

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
                    Write-Ui "  1. Forex Trading (Walk-Forward Curriculum Orchestrator)" -Color Cyan
                    Write-Ui "  B. Back to Domain Menu" -Color Gray
                    Write-Ui ""

                    $subChoice = (Read-Host "Select sub-category (1, B)").Trim()
                    if ($subChoice -eq 'B' -or $subChoice -eq 'b') { break }

                    $modelList = @()
                    switch ($subChoice) {
                        '1' { $modelList = @("forex_curriculum") }
                        default { continue }
                    }

                    $selected = Show-ModelList -Models $modelList
                    if ($null -ne $selected) { return $selected }
                }
            }

            default {
                Write-Ui "Invalid selection." -Color Red
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
        Write-Ui "  $($i+1). $($Models[$i])" -Color Green
    }
    Write-Ui "  $($Models.Count + 1). Back" -Color Gray
    Write-Ui ""

    $choice = (Read-Host "Select a model (1-$($Models.Count + 1))").Trim()
    if ($choice -as [int] -and [int]$choice -ge 1 -and [int]$choice -le $Models.Count) {
        return $Models[[int]$choice - 1]
    }
    return $null
}

function Invoke-KaggleCloudMenu {
    while ($true) {
        Write-Header "KAGGLE CLOUD ENGINE (HEADLESS GPU ORCHESTRATION)"
        Write-Ui "  1. Train on Kaggle            (Launch notebook, stream logs, auto-pull checkpoints)" -Color Cyan
        Write-Ui "  2. Monitor Active Cloud Jobs   (Monitor only: stream live logs, no pulling)" -Color Cyan
        Write-Ui "  3. Pull & Save Checkpoints     (Downloads .pth & metrics.csv to local disk)" -Color Cyan
        Write-Ui "  4. Setup / Verify Credentials  (Configure Kaggle Username & Token)" -Color Cyan
        Write-Ui "  B. Back to Main Menu" -Color Gray
        Write-Ui ""

        $cloudChoice = (Read-Host "Select cloud option (1-4, B)").Trim()
        if ($cloudChoice -eq 'B' -or $cloudChoice -eq 'b') { return }

        switch ($cloudChoice) {
            '1' {
                $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" -m training.kaggle_monitor --action train
                Pop-Location
                Read-Host "Press Enter to return..."
            }
            '2' {
                $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" -m training.kaggle_monitor --action monitor
                Pop-Location
                Read-Host "Press Enter to return..."
            }
            '3' {
                $targetModel = Get-ModelSelection
                if ($null -ne $targetModel) {
                    $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""
                    $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                    Push-Location $script:HUB_DIR
                    & "$script:VENV_DIR\Scripts\python.exe" -m training.kaggle_cloud_manager --action pull --model $targetModel
                    Pop-Location
                }
                Read-Host "Press Enter to return..."
            }
            '4' {
                $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR
                & "$script:VENV_DIR\Scripts\python.exe" -m training.kaggle_cloud_manager --action setup_auth
                Pop-Location
                Read-Host "Press Enter to return..."
            }
            default { Write-Ui "Invalid selection."; Start-Sleep -Seconds 1 }
        }
    }
}

function Show-Menu {
    Clear-Host
    Write-Header "LEMGENDARY AI TRAINING SUITE (2026 SPECIALIZATION)"
    Write-Ui " [ENVIRONMENT: $(if ($env:VIRTUAL_ENV) { 'VIRTUAL' } else { 'GLOBAL' })]" -Color Gray
    Write-Ui "  1. Train Individual Model      (Launches Local LemGendary Training Suite)"
    Write-Ui "  2. Single-Epoch Unit Test      (Diagnostic 1-Epoch pass for ALL models)"
    Write-Ui "  3. Kaggle Cloud Engine         (Headless GPU: Launch / Monitor / Pull)"
    Write-Ui "  Q. Exit"
    Write-Ui ""
}

# Pre-Flight Bootstrap Python check (Silent if found)
Invoke-BootstrapCheck

while ($true) {
    Show-Menu
    $choice = (Read-Host "Select an option (1-3, Q)").Trim()
    switch ($choice) {
        '1' {
            if (Test-Environment) {
                $selectedModel = Get-ModelSelection
                if ($null -ne $selectedModel) {
                    $extraArgs = @()
                    Write-Ui "  [*] Launching Training Matrix for >> $selectedModel <<..." -Color Green
                    Write-Ui "      -> Target Manifold: $selectedModel" -Color Gray
                    Invoke-JanitorPurge # Ensure clean start
                    $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                    $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                    if ($selectedModel -eq "forex_curriculum") {
                        Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" -m training.train_forex_curriculum; Pop-Location
                    } else {
                        Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" -m training.train --model $selectedModel $extraArgs; Pop-Location
                    }
                }
            }
            Read-Host "Press Enter to return..."
        }
        '2' {
            if (Test-Environment) {
                Invoke-JanitorPurge
                $env:PYTHONPATH=""; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "train_all.py" --epochs 1 --yes; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '3' {
            if (Test-Environment) {
                Invoke-KaggleCloudMenu
            }
        }
        'Q' { return }
        'q' { return }
        default { Write-Ui "Invalid selection."; Start-Sleep -Seconds 1 }
    }
}
