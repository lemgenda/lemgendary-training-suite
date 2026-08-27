# training/lemgendary_hub.ps1 [Refresh: 2026-08-18_11:15]
# Master Orchestration Script for LemGendary AI Training & Management
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null


$script:HUB_DIR = $PSScriptRoot
if (-not $script:HUB_DIR) { $script:HUB_DIR = Get-Location }

# Load Infrastructure Logic
. (Join-Path $script:HUB_DIR "lemgendary_env_manager.ps1")

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
                # --- DOMAIN 1: Image Manipulation & Restoration ---
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
                # --- DOMAIN 2: Image Generation & Multimodal ---
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
                    Write-Host "  1. Forex Trading (Walk-Forward Curriculum Orchestrator)" -ForegroundColor Cyan
                    Write-Host "  B. Back to Domain Menu" -ForegroundColor Gray
                    Write-Host ""

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

function Invoke-KaggleCloudMenu {
    while ($true) {
        Write-Header "KAGGLE CLOUD ENGINE (HEADLESS GPU ORCHESTRATION)"
        Write-Host "  1. Launch Cloud Training Job  (Pushes & executes kernel on Tesla T4/P100)" -ForegroundColor Cyan
        Write-Host "  2. Monitor Active Cloud Jobs   (Live terminal status & log streaming)" -ForegroundColor Cyan
        Write-Host "  3. Pull & Save Checkpoints     (Downloads .pth & metrics.csv to local disk)" -ForegroundColor Cyan
        Write-Host "  4. Setup / Verify Credentials  (Configure Kaggle Username & Token)" -ForegroundColor Cyan
        Write-Host "  B. Back to Main Menu" -ForegroundColor Gray
        Write-Host ""

        $cloudChoice = (Read-Host "Select cloud option (1-4, B)").Trim()
        if ($cloudChoice -eq 'B' -or $cloudChoice -eq 'b') { return }

        switch ($cloudChoice) {
            '1' {
                $targetModel = Get-ModelSelection
                if ($null -ne $targetModel) {
                    Write-Host "  [CLOUD] Deploying GPU training for >> $targetModel <<..." -ForegroundColor Green
                    $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""
                    $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                    Push-Location $script:HUB_DIR
                    & "$script:VENV_DIR\Scripts\python.exe" -m training.kaggle_cloud_manager --action launch --model $targetModel
                    Pop-Location
                }
                Read-Host "Press Enter to return..."
            }
            '2' {
                $targetModel = Get-ModelSelection
                if ($null -ne $targetModel) {
                    $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""
                    $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                    Push-Location $script:HUB_DIR
                    & "$script:VENV_DIR\Scripts\python.exe" -m training.kaggle_cloud_manager --action monitor --model $targetModel
                    Pop-Location
                }
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
            default { Write-Host "Invalid selection."; Start-Sleep -Seconds 1 }
        }
    }
}

function Show-Menu {
    Clear-Host
    Write-Header "LEMGENDARY AI TRAINING SUITE (2026 SPECIALIZATION)"
    Write-Host " [ENVIRONMENT: $(if ($env:VIRTUAL_ENV) { 'VIRTUAL' } else { 'GLOBAL' })]" -ForegroundColor Gray
    Write-Host "  1. Initialize/Fix All Systems (Python + Node.js + Specialized GPUs)"
    Write-Host "  2. Train Individual Model      (Launches Local LemGendary Training Suite)"
    Write-Host "  3. Single-Epoch Unit Test      (Diagnostic 1-Epoch pass for ALL models)"
    Write-Host "  4. Kaggle Cloud Engine         (Headless GPU: Launch / Monitor / Pull)"
    Write-Host "  Q. Exit"
    Write-Host ""
}

# Pre-Flight Bootstrap Python check (Silent if found)
Invoke-BootstrapCheck

while ($true) {
    Show-Menu
    $choice = (Read-Host "Select an option (1-4, Q)").Trim()
    switch ($choice) {
        '1' { Initialize-Environment; Read-Host "Press Enter to return..." }
        '2' {
            if (Test-Environment) {
                $selectedModel = Get-ModelSelection
                if ($null -ne $selectedModel) {
                    $extraArgs = @()
                    Write-Host "  [*] Launching Training Matrix for >> $selectedModel <<..." -ForegroundColor Green
                    Write-Host "      -> Target Manifold: $selectedModel" -ForegroundColor Gray
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
        '3' {
            if (Test-Environment) {
                Invoke-JanitorPurge
                $env:PYTHONPATH=""; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" "train_all.py" --epochs 1 --yes; Pop-Location
            }
            Read-Host "Press Enter to return..."
        }
        '4' {
            if (Test-Environment) {
                Invoke-KaggleCloudMenu
            }
        }
        'Q' { return }
        'q' { return }
        default { Write-Host "Invalid selection."; Start-Sleep -Seconds 1 }
    }
}
