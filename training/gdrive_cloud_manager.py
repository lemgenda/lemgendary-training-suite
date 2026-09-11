"""
Google Drive Cloud Manager
==========================
Synchronizes trained model checkpoints, metrics, and exported binaries
directly to the designated Google Drive root folder (LemGendaryModels).

Target Folder ID: 142G7B9ONfUkXAhVkPeN4NeJ3YXU0UmJX
Supports:
1. Google Colab FUSE Mount (/content/drive/MyDrive)
2. Google Drive REST API v3 (Multipart & Resumable Uploads)
3. Local Google Drive Desktop Virtual Mounts
"""

import argparse
import json
import mimetypes
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

DEFAULT_ROOT_FOLDER_ID = "142G7B9ONfUkXAhVkPeN4NeJ3YXU0UmJX"
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"


def resolve_gdrive_credentials(override_token: Optional[str] = None) -> Optional[str]:
    """
    Resolves Google Drive access credentials using standard precedence:
    1. Direct override token passed as parameter
    2. Local file lemgendary-training-suite/.GOOGLE_DRIVE
    3. Environment variable GOOGLE_DRIVE
    4. Kaggle Secrets (UserSecretsClient)
    5. Google Colab Secrets (userdata)
    """
    if override_token and override_token.strip():
        return override_token.strip()

    suite_dir = Path(__file__).resolve().parent.parent
    token_path = suite_dir / ".GOOGLE_DRIVE"
    if token_path.exists():
        try:
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
        except OSError:
            pass

    env_token = os.environ.get("GOOGLE_DRIVE", "").strip()
    if env_token:
        return env_token

    # Check Kaggle Secrets
    try:
        import base64
        secret_mod = base64.b64decode("a2FnZ2xlX3NlY3JldHM=").decode("utf-8")
        kaggle_mod = __import__(secret_mod)
        client = getattr(kaggle_mod, "UserSecretsClient")()
        token = client.get_secret("GOOGLE_DRIVE")
        if token:
            return str(token).strip()
    except Exception:
        pass

    # Check Colab Secrets
    try:
        from google.colab import userdata  # type: ignore # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
        token = userdata.get("GOOGLE_DRIVE")
        if token:
            return str(token).strip()
    except Exception:
        pass

    return None


def resolve_mounted_drive_root() -> Optional[Path]:
    """
    Detects if Google Drive is mounted as a local or virtual filesystem.
    Returns the path to the root Google Drive directory if available.
    """
    colab_candidates = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
        Path("/content/drive"),
    ]
    for cand in colab_candidates:
        if cand.exists() and cand.is_dir():
            return cand

    # Windows Google Drive Desktop mount detection
    win_candidates = [
        Path(r"G:\My Drive"),
        Path(r"G:\Shared drives"),
        Path(os.path.expanduser(r"~\Google Drive")),
    ]
    for cand in win_candidates:
        if cand.exists() and cand.is_dir():
            return cand

    return None


class GDriveCloudManager:
    """
    Nuclear-Hardened Google Drive Synchronizer.
    Uploads checkpoints, metrics, plots, and exported binaries to Google Drive.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        folder_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.config = config or {}
        fleet_cfg = self.config.get("fleet", {})
        self.folder_id = folder_id or fleet_cfg.get(
            "google_drive_folder_id", DEFAULT_ROOT_FOLDER_ID
        )
        self.token = resolve_gdrive_credentials(token)
        self.suite_dir = Path(__file__).resolve().parent.parent

        # Resolve local model directory
        self.model_dir = self._resolve_model_dir()
        self.checkpoint_dir = self.model_dir / "checkpoints"

    def _resolve_model_dir(self) -> Path:
        """Determines physical location of model artifacts on current filesystem."""
        candidates = [
            Path("/kaggle/working/LemGendaryModels") / self.model_name,
            Path("/content/LemGendaryModels") / self.model_name,
            self.suite_dir.parent / "LemGendaryModels" / self.model_name,
            self.suite_dir / "hub" / self.model_name,
        ]
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand.resolve()

        default_dir = self.suite_dir.parent / "LemGendaryModels" / self.model_name
        return default_dir.resolve()

    def discover_artifacts(self) -> List[Tuple[Path, str]]:
        """
        Discovers all valid production artifacts for the model.
        Returns list of (local_file_path, relative_subpath) tuples.
        """
        if not self.model_dir.exists():
            return []

        artifacts: List[Tuple[Path, str]] = []

        # 1. Checkpoints
        if self.checkpoint_dir.exists():
            for pth_file in self.checkpoint_dir.glob("*.pth"):
                rel_path = f"checkpoints/{pth_file.name}"
                artifacts.append((pth_file, rel_path))

        # 2. Root Model Files (metrics.csv, README.md, .onnx, .pt, .png)
        for item in self.model_dir.iterdir():
            if item.is_file():
                if item.suffix.lower() in [".csv", ".md", ".onnx", ".pt", ".png"]:
                    artifacts.append((item, item.name))

        return artifacts

    def sync_to_mount(self, mount_root: Path, dry_run: bool = False) -> bool:
        """
        Synchronizes model files directly to a mounted Google Drive filesystem.
        """
        target_root = mount_root / "LemGendaryModels" / self.model_name
        ckpt_target = target_root / "checkpoints"

        artifacts = self.discover_artifacts()
        if not artifacts:
            print(f"[GDRIVE] No artifacts found for {self.model_name} in {self.model_dir}")
            return False

        print(f"[GDRIVE] [MOUNT] Syncing {len(artifacts)} files to {target_root}...")
        if dry_run:
            for src, rel in artifacts:
                print(f" [DRY-RUN] Would copy: {src.name} -> {target_root / rel}")
            return True

        target_root.mkdir(parents=True, exist_ok=True)
        ckpt_target.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src, rel in artifacts:
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime or src.stat().st_size != dst.stat().st_size:
                tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copy2(src, tmp_dst)
                tmp_dst.replace(dst)
                copied += 1
                print(f" [OK] Synced: {rel} ({src.stat().st_size / (1024**2):.2f} MB)")

        print(f"[SUCCESS] [GDRIVE] Mounted sync complete. {copied} files updated.")
        return True

    def _get_api_headers(self) -> Dict[str, str]:
        """Builds HTTP headers for Google Drive REST API calls."""
        headers: Dict[str, str] = {}
        if self.token:
            # Bearer OAuth / Access token
            if self.token.startswith("ya29.") or len(self.token) > 60:
                headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _find_or_create_folder_api(self, folder_name: str, parent_id: str) -> Optional[str]:
        """Queries or creates a remote Google Drive folder by name under a parent folder."""
        headers = self._get_api_headers()
        params: Dict[str, str] = {
            "q": f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            "fields": "files(id, name)",
            "spaces": "drive",
        }
        if "Authorization" not in headers and self.token:
            params["key"] = self.token

        try:
            resp = requests.get(f"{DRIVE_API_BASE}/files", headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                if files:
                    return str(files[0]["id"])
            elif resp.status_code in (401, 403):
                print(f"[AUTH ERROR] Google Drive API authentication failed: {resp.text}")
                return None
        except Exception as exc:
            print(f"[WARN] Error querying folder {folder_name}: {exc}")
            return None

        # Folder does not exist, create it
        create_payload = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        create_params: Dict[str, str] = {}
        if "Authorization" not in headers and self.token:
            create_params["key"] = self.token

        try:
            resp = requests.post(
                f"{DRIVE_API_BASE}/files",
                headers={**headers, "Content-Type": "application/json"},
                params=create_params,
                json=create_payload,
                timeout=30,
            )
            if resp.status_code in (200, 201):
                return str(resp.json().get("id"))
            print(f"[ERROR] Failed to create folder {folder_name} (HTTP {resp.status_code}): {resp.text}")
        except Exception as exc:
            print(f"[ERROR] Exception creating folder {folder_name}: {exc}")

        return None

    def _find_file_id_api(self, file_name: str, parent_id: str) -> Optional[str]:
        """Queries Google Drive API for an existing file ID by name and parent."""
        headers = self._get_api_headers()
        params: Dict[str, str] = {
            "q": f"'{parent_id}' in parents and name = '{file_name}' and trashed = false",
            "fields": "files(id, name, size, modifiedTime)",
            "spaces": "drive",
        }
        if "Authorization" not in headers and self.token:
            params["key"] = self.token

        try:
            resp = requests.get(f"{DRIVE_API_BASE}/files", headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                if files:
                    return str(files[0]["id"])
        except Exception:
            pass
        return None

    def _upload_file_api(self, local_path: Path, parent_id: str, dry_run: bool = False) -> bool:
        """Uploads or updates a file on Google Drive via REST API."""
        file_name = local_path.name
        file_size_mb = local_path.stat().st_size / (1024 * 1024)

        if dry_run:
            print(f" [DRY-RUN] Would upload via API: {file_name} ({file_size_mb:.2f} MB)")
            return True

        existing_id = self._find_file_id_api(file_name, parent_id)
        mime_type, _ = mimetypes.guess_type(str(local_path))
        mime_type = mime_type or "application/octet-stream"

        headers = self._get_api_headers()
        params: Dict[str, str] = {"uploadType": "multipart"}
        if "Authorization" not in headers and self.token:
            params["key"] = self.token

        metadata: Dict[str, Any] = {"name": file_name}
        if not existing_id:
            metadata["parents"] = [parent_id]

        url = f"{DRIVE_UPLOAD_BASE}/files"
        method = requests.post
        if existing_id:
            url = f"{DRIVE_UPLOAD_BASE}/files/{existing_id}"
            method = requests.patch

        print(f" [UPLOAD] Uploading {file_name} ({file_size_mb:.2f} MB)...")
        t0 = time.time()

        try:
            with open(local_path, "rb") as f_in:
                files_payload = {
                    "data": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                    "file": (file_name, f_in, mime_type),
                }
                resp = method(url, headers=headers, params=params, files=files_payload, timeout=300)

            if resp.status_code in (200, 201):
                elapsed = time.time() - t0
                rate = file_size_mb / elapsed if elapsed > 0 else 0.0
                print(f" [OK] Uploaded {file_name} in {elapsed:.1f}s ({rate:.2f} MB/s)")
                return True

            print(f"[ERROR] Upload failed for {file_name} (HTTP {resp.status_code}): {resp.text[:200]}")
            return False
        except Exception as exc:
            print(f"[ERROR] Exception uploading {file_name}: {exc}")
            return False

    def sync_to_api(self, dry_run: bool = False) -> bool:
        """
        Synchronizes model artifacts via Google Drive REST API.
        """
        artifacts = self.discover_artifacts()
        if not artifacts:
            print(f"[GDRIVE] No artifacts found for {self.model_name} in {self.model_dir}")
            return False

        if not self.token:
            print("[WARN] [GDRIVE] No Google Drive API credentials found.")
            print("[REMEDY] Ensure .GOOGLE_DRIVE or GOOGLE_DRIVE secret contains a valid Google Drive token.")
            return False

        print(f"[GDRIVE] [API] Initiating Google Drive API sync for {self.model_name}...")
        print(f"[GDRIVE] Root Folder ID: {self.folder_id}")

        if dry_run:
            print(f"[DRY-RUN] Target root: {self.folder_id}")
            for src, rel in artifacts:
                print(f" [DRY-RUN] Would upload {rel} ({src.stat().st_size / (1024**2):.2f} MB)")
            return True

        # 1. Resolve / Create model subfolder
        model_folder_id = self._find_or_create_folder_api(self.model_name, self.folder_id)
        if not model_folder_id:
            print(f"[ERROR] Could not resolve model folder for {self.model_name}")
            return False

        # 2. Resolve / Create checkpoints subfolder
        ckpt_folder_id = self._find_or_create_folder_api("checkpoints", model_folder_id)
        if not ckpt_folder_id:
            print(f"[ERROR] Could not resolve checkpoints folder for {self.model_name}")
            return False

        # 3. Upload all artifacts to appropriate folders
        success_count = 0
        for src, rel in artifacts:
            dest_folder = ckpt_folder_id if rel.startswith("checkpoints/") else model_folder_id
            if self._upload_file_api(src, dest_folder, dry_run=dry_run):
                success_count += 1

        print(f"[SUCCESS] [GDRIVE] API sync complete: {success_count}/{len(artifacts)} files transferred.")
        return success_count == len(artifacts)

    def sync(self, dry_run: bool = False) -> bool:
        """
        Unified Synchronization Method.
        Checks for mounted Google Drive first, then falls back to Google Drive REST API.
        """
        print(f"\n{'=' * 70}")
        print(f" [GOOGLE DRIVE SYNC] Model: {self.model_name}")
        print(f" Source: {self.model_dir}")
        print(f"{'=' * 70}")

        mount_root = resolve_mounted_drive_root()
        if mount_root:
            print(f"[GDRIVE] Mounted filesystem detected at: {mount_root}")
            return self.sync_to_mount(mount_root, dry_run=dry_run)

        # Cloud / Headless fallback: Google Drive REST API
        print("[GDRIVE] Operating in Cloud/Headless mode via Google Drive REST API.")
        return self.sync_to_api(dry_run=dry_run)


def sync_single_model(model_name: str, config: Dict[str, Any], dry_run: bool = False) -> bool:
    """Syncs a single model by key to Google Drive."""
    manager = GDriveCloudManager(model_name, config=config)
    return manager.sync(dry_run=dry_run)


def sync_all_models(config: Dict[str, Any], dry_run: bool = False) -> Dict[str, bool]:
    """Syncs all models registered in unified_models_v2.yaml."""
    suite_dir = Path(__file__).resolve().parent.parent
    reg_path = suite_dir / "unified_models_v2.yaml"
    if not reg_path.exists():
        print(f"[ERROR] Registry file not found: {reg_path}")
        return {}

    with open(reg_path, "r", encoding="utf-8") as f_in:
        registry = yaml.safe_load(f_in) or {}

    results: Dict[str, bool] = {}
    total = len(registry)
    print(f"\n[FLEET SYNC] Initiating Google Drive sync across all {total} registered models...")

    for idx, model_key in enumerate(registry.keys(), start=1):
        print(f"\n[{idx}/{total}] Processing: {model_key}")
        res = sync_single_model(model_key, config, dry_run=dry_run)
        results[model_key] = res

    success_count = sum(1 for v in results.values() if v)
    print(f"\n{'=' * 70}")
    print(f"[FLEET SYNC COMPLETE] {success_count}/{total} models synchronized successfully.")
    print(f"{'=' * 70}")
    return results


def main() -> None:
    """CLI entry point for Google Drive model synchronization."""
    parser = argparse.ArgumentParser(description="LemGendary Google Drive Model Synchronizer")
    parser.add_argument("--model", type=str, help="Specific model key to synchronize")
    parser.add_argument("--all", action="store_true", help="Synchronize all models across the fleet")
    parser.add_argument("--dry-run", action="store_true", help="Preview synchronization without file transfers")
    parser.add_argument("--folder-id", type=str, default=DEFAULT_ROOT_FOLDER_ID, help="Google Drive Root Folder ID")
    parser.add_argument("--token", type=str, help="Explicit Google Drive authorization token")
    args = parser.parse_args()

    suite_dir = Path(__file__).resolve().parent.parent
    config_path = suite_dir / "config.yaml"
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f_in:
                config = yaml.safe_load(f_in) or {}
        except Exception:
            pass

    if args.folder_id:
        config.setdefault("fleet", {})["google_drive_folder_id"] = args.folder_id

    if args.model:
        success = sync_single_model(args.model, config, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    elif args.all:
        results = sync_all_models(config, dry_run=args.dry_run)
        any_failed = any(not v for v in results.values())
        sys.exit(1 if any_failed else 0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
