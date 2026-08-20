import os
import sys
import argparse
import subprocess
import glob

def publish_model_release(model_key, models_root, tag="v1.0.0-sota", repo="lemgenda/lemgendary-pretrained-models"):
    """
    Publishes all compiled binary assets for a given model to a GitHub Release.
    """
    model_dir = os.path.join(models_root, model_key)
    if not os.path.exists(model_dir):
        print(f"[ERROR] Model directory not found: {model_dir}")
        return False
        
    # Discover all weight binaries
    extensions = ["*.onnx", "*.pt", "*.pth", "*.data", "*.safetensors"]
    asset_files = []
    for ext in extensions:
        asset_files.extend(glob.glob(os.path.join(model_dir, ext)))
        
    if not asset_files:
        print(f"[WARNING] No binary assets found in {model_dir}")
        return False
        
    print(f"\n[RELEASE] Model: {model_key} | Tag: {tag} | Assets ({len(asset_files)}):")
    for a in asset_files:
        size_mb = os.path.getsize(a) / (1024 * 1024)
        print(f"  -> {os.path.basename(a)} ({size_mb:.2f} MB)")
        
    # Check if release exists
    check_cmd = ["gh", "release", "view", tag, "--repo", repo]
    res = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        # Create the release
        print(f"[RELEASE] Creating new GitHub Release: {tag} in {repo}...")
        create_cmd = [
            "gh", "release", "create", tag,
            "--repo", repo,
            "--title", f"LemGendary SOTA Model Weights ({tag})",
            "--notes", f"Official production weights and ONNX exports for LemGendary AI models."
        ]
        res_create = subprocess.run(create_cmd, capture_output=True, text=True)
        if res_create.returncode != 0:
            print(f"[ERROR] Failed to create release: {res_create.stderr}")
            return False
        print(f"[OK] Release {tag} created successfully.")
        
    # Upload assets
    upload_cmd = ["gh", "release", "upload", tag, "--repo", repo, "--clobber"] + asset_files
    print(f"[RELEASE] Uploading {len(asset_files)} assets to {tag}...")
    res_upload = subprocess.run(upload_cmd, capture_output=True, text=True)
    if res_upload.returncode == 0:
        print(f"[SUCCESS] Assets for {model_key} successfully uploaded to {repo}@{tag}!")
        return True
    else:
        print(f"[ERROR] Asset upload failed: {res_upload.stderr}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LemGendary GitHub Releases Publisher")
    parser.add_argument("--model", type=str, help="Publish binary weights for a specific model key.")
    parser.add_argument("--all", action="store_true", help="Publish binary weights for all available models.")
    parser.add_argument("--tag", type=str, default="v1.0.0-sota", help="GitHub Release tag (default: v1.0.0-sota).")
    parser.add_argument("--repo", type=str, default="lemgenda/lemgendary-pretrained-models", help="GitHub repository.")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_root = os.path.abspath(os.path.join(base_dir, "../LemGendaryModels"))
    
    if args.all:
        model_dirs = sorted([d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d)) and not d.startswith('.')])
        print(f"[START] Publishing weights for {len(model_dirs)} models to {args.repo} ({args.tag})...")
        success_count = 0
        for m in model_dirs:
            if publish_model_release(m, models_root, tag=args.tag, repo=args.repo):
                success_count += 1
        print(f"\n[DONE] Successfully published {success_count}/{len(model_dirs)} models to GitHub Releases.")
    elif args.model:
        publish_model_release(args.model, models_root, tag=args.tag, repo=args.repo)
    else:
        parser.print_help()
