# training/lemgendary_hub.ps1 [Refresh: 2026-03-27_23:50]
# Master Orchestration Script for LemGendary AI Training & Management
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null


$script:HUB_DIR = $PSScriptRoot
if (-not $script:HUB_DIR) { $script:HUB_DIR = Get-Location }

# PowerShell 5.1 compatibility for Join-Path

# Load Infrastructure Logic
. (Join-Path $script:HUB_DIR "lemgendary_env_manager.ps1")

function Get-ModelSelection {
    Write-Header "SELECT MODEL CATEGORY"
    Write-Host "  1. Quality Assessment  (NIMA, Aesthetics, Authenticity)" -ForegroundColor Cyan
    Write-Host "  2. Face & Detection    (RetinaFace, YOLOv8, CodeFormer, ParseNet)" -ForegroundColor Cyan
    Write-Host "  3. Super-Resolution    (UltraZoom x2/x3/x4/x8)" -ForegroundColor Cyan
    Write-Host "  4. Image Restoration   (NAFNet, MIRNet, FFANet, MPRNet)" -ForegroundColor Cyan
    Write-Host "  5. Universal Hybrid    (UPN v2, Multi-Restorer, Film Restoration)" -ForegroundColor Cyan
    Write-Host "  6. Generative Diffusion (Master Manifold)" -ForegroundColor Cyan
    Write-Host "  7. Vision-Language      (Master Manifold)" -ForegroundColor Cyan
    Write-Host "  8. Cancel" -ForegroundColor Gray
    Write-Host ""
    
    $catChoice = Read-Host "Select a category (1-8)"
    $modelList = @()
    switch ($catChoice) {
        '1' { $modelList = @("nima_aesthetic_mobile", "nima_aesthetic_efficientnet", "nima_aesthetic_pro", "nima_technical", "nima_authenticity", "anime_nsfw_classification") }
        '2' { $modelList = @("codeformer", "parsenet", "retinaface_mobilenet", "retinaface_resnet", "yolov8n") }
        '3' { $modelList = @("ultrazoom") }
        '4' { $modelList = @("ffanet_indoor", "ffanet_outdoor", "mprnet_deraining", "mirnet_lowlight", "mirnet_exposure", "nafnet_debluring", "nafnet_denoising") }
        '5' { $modelList = @("upn_v2", "professional_multitask_restoration", "film_restorer") }
        '6' { $modelList = @("diffusion_sdxl", "diffusion_flux") }
        '7' { $modelList = @("vlm_llava", "vlm_blip2") }
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


function Show-Menu {
    Clear-Host
    Write-Header "LEMGENDARY AI TRAINING SUITE (2026 SPECIALIZATION)"
    Write-Host " [ENVIRONMENT: $(if ($env:VIRTUAL_ENV) { 'VIRTUAL' } else { 'GLOBAL' })]" -ForegroundColor Gray
    Write-Host "  1. Initialize/Fix All Systems (Python + Node.js + Specialized GPUs)"
    Write-Host "  2. Train Individual Model      (Launches LemGendary Training Suite)"
    Write-Host "  3. Single-Epoch Unit Test      (Diagnostic 1-Epoch pass for ALL models)"
    Write-Host "  Q. Exit"
    Write-Host ""
}

# Pre-Flight Bootstrap Python check (Silent if found)
Invoke-BootstrapCheck

while ($true) {
    Show-Menu
    $choice = (Read-Host "Select an option (1-3, Q)").Trim()
    switch ($choice) {
        '1' { Initialize-Environment; Read-Host "Press Enter to return..." }
        '2' {
            if (Test-Environment) {
                $selectedModel = Get-ModelSelection
                if ($null -ne $selectedModel) {
                    $rocket = [char]0xD83D + [char]0xDE80
                    $extraArgs = @()
                    Write-Host "  [$rocket] Launching Training Matrix for >> $selectedModel <<..." -ForegroundColor Green
                    Write-Host "      -> Target Manifold: $selectedModel" -ForegroundColor Gray
                    Invoke-JanitorPurge # Ensure clean start
                    $env:PYTHONPATH="$script:HUB_DIR"; $env:PYTHONHOME=""; $env:TRITON_SILENT="1"
                    $env:PATH="$script:VENV_DIR\Scripts;$script:VENV_DIR\bin;$env:PATH"
                    Push-Location $script:HUB_DIR; & "$script:VENV_DIR\Scripts\python.exe" -m training.train --model $selectedModel $extraArgs; Pop-Location
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
        'Q' { return }
        'q' { return }
        default { Write-Host "Invalid selection."; Start-Sleep -Seconds 1 }
    }
}
