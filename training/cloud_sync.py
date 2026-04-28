import os
import threading
import zipfile
import requests
from pathlib import Path

def _sync_worker(model_name, epoch, config):
    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        return
        
    print(f"\n☁️ [CLOUD SYNC] Initiating async artifact push for {model_name} (Epoch {epoch})...")
    repo = "lemgenda/ai-models"
    tag = "kaggle-latest"
    
    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Zip the artifacts
    zip_name = f"{model_name}_artifacts.zip"
    zip_path = Path("/tmp") / zip_name if os.name != 'nt' else Path(os.environ.get("TEMP", "C:/Windows/Temp")) / zip_name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add best.pth from internal checkpoints fallback
        ckpt_dir = Path(config.get("checkpoint_dir", "trained-models/checkpoints"))
        best_pth = ckpt_dir / f"{model_name}_best.pth"
        if best_pth.exists():
            zipf.write(best_pth, best_pth.name)
            
        # 2026 Shift: Zip from decoupled LemGendaryModels root volume
        project_root = Path(__file__).resolve().parent.parent
        export_base = config.get("export_dir", "../LemGendaryModels")
        # Ensure we accurately target the specific model's output envelope
        export_dir = (project_root / export_base / model_name).resolve()
        
        if export_dir.exists():
            for root, dirs, files in os.walk(export_dir):
                for f in files:
                    fp = Path(root) / f
                    # Add all compiled .onnx, metrics.csv, README, and external checkpoints
                    zipf.write(fp, str(fp.relative_to(export_dir)))
                    
    # 2. Get Release ID or Create Release
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    r = requests.get(release_url, headers=headers)
    
    if r.status_code == 404:
        # Create it
        post_url = f"https://api.github.com/repos/{repo}/releases"
        payload = {"tag_name": tag, "name": "Latest Kaggle Artifacts", "body": "Auto-pushed from Kaggle Cloud Training."}
        r = requests.post(post_url, headers=headers, json=payload)
        
    if r.status_code not in (200, 201):
        print(f"☁️ [CLOUD SYNC] Failed to resolve release: {r.text}")
        return
        
    release_data = r.json()
    release_id = release_data["id"]
    upload_url = release_data["upload_url"].split("{")[0]
    
    # 3. Delete existing asset if it exists
    for asset in release_data.get("assets", []):
        if asset["name"] == zip_name:
            requests.delete(asset["url"], headers=headers)
            
    # 4. Upload new asset
    upload_headers = headers.copy()
    upload_headers["Content-Type"] = "application/zip"
    
    print(f"☁️ [CLOUD SYNC] Pushing {zip_path.stat().st_size / (1024*1024):.1f} MB to GitHub Releases...")
    with open(zip_path, 'rb') as f:
        r = requests.post(f"{upload_url}?name={zip_name}", headers=upload_headers, data=f)
        
    if r.status_code == 201:
        print(f"✅ [CLOUD SYNC] Artifacts for {model_name} successfully uploaded to GitHub!")
    else:
        print(f"❌ [CLOUD SYNC] Upload failed: {r.status_code} - {r.text}")
        
    # Cleanup
    if zip_path.exists():
        zip_path.unlink()

def trigger_cloud_sync(model_name, epoch, config):
    if not os.environ.get("GITHUB_PAT"):
        return
        
    t = threading.Thread(target=_sync_worker, args=(model_name, epoch, config), daemon=True)
    t.start()
