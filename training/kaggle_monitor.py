import os
import sys
import re
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

# --- Path Anchor Defense ---
# Anchor the search path to the workspace root and script directory to ensure module discovery
_script_dir = Path(__file__).resolve().parent
_suite_dir = _script_dir.parent
_workspace_root = _suite_dir.parent
if str(_suite_dir) not in sys.path:
    sys.path.insert(0, str(_suite_dir))
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from training.kaggle_cloud_manager import (
    create_cloud_kernel_bundle,
    get_kernel_slug,
    pull_kaggle_artifacts,
)

def find_kaggle_users_file(explicit_path: Optional[Path] = None) -> Optional[Path]:
    """
    Locates the .kaggle_users credentials registry file.
    Searches explicit path, local suite directory, working directory, and sibling directories.
    """
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path.cwd() / ".kaggle_users",
        Path(__file__).resolve().parent.parent / ".kaggle_users",
        Path.cwd().parent / "lemgendary-training-suite" / ".kaggle_users",
        Path(__file__).resolve().parent.parent.parent / "lemgendary-training-suite" / ".kaggle_users",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return Path(__file__).resolve().parent.parent / ".kaggle_users"


def load_kaggle_users(users_file: Optional[Path] = None) -> Tuple[List[Dict[str, str]], Path]:
    """
    Parses configured Kaggle user accounts from .kaggle_users.
    Expected format per line: KAGGLE_USERNAME=<user>, KAGGLE_API_TOKEN=<token>;
    """
    target_file = find_kaggle_users_file(users_file)
    accounts: List[Dict[str, str]] = []

    if target_file and target_file.exists():
        try:
            content = target_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m_user = re.search(r"KAGGLE_USERNAME=([^,;\s]+)", line)
                m_token = re.search(r"KAGGLE_API_TOKEN=([^,;\s]+)", line)
                if m_user and m_token:
                    accounts.append({
                        "username": m_user.group(1).strip(),
                        "token": m_token.group(1).strip()
                    })
        except OSError as e:
            print(f"[WARN] Failed to read users registry at {target_file}: {e}")

    return accounts, target_file or (Path.cwd() / ".kaggle_users")


def save_kaggle_user(users_file: Path, username: str, token: str) -> bool:
    """
    Appends a new user account definition to the .kaggle_users file.
    """
    try:
        users_file.parent.mkdir(parents=True, exist_ok=True)
        with open(users_file, "a", encoding="utf-8") as f:
            f.write(f"\nKAGGLE_USERNAME={username}, KAGGLE_API_TOKEN={token};")
        return True
    except OSError as e:
        print(f"[ERROR] Failed to save user to {users_file}: {e}")
        return False


def authenticate_kaggle_user(username: str, token: str) -> Optional[Any]:
    """
    Configures environment variables and initializes authenticated KaggleApi instance.
    """
    clean_token = token.replace("KGAT_", "").strip().rstrip(";").strip()

    # Clear conflicting access token variables that trigger OAuth introspection
    os.environ.pop("KAGGLE_API_TOKEN", None)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = clean_token

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:
        print(f"\n[ERROR] Authentication failed for user '{username}': {exc}")
        print("[REMEDY] Verify username and API token validity on kaggle.com/settings/api.")
        return None


def fetch_user_kernels(api: Any, username: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Queries active and recent kernels for the target Kaggle user with concurrent status resolution.
    """
    try:
        raw_kernels = api.kernels_list(user=username, page_size=limit, sort_by="dateRun")
    except Exception as exc:
        print(f"[ERROR] Failed to list kernels for {username}: {exc}")
        return []

    if not raw_kernels:
        return []

    def _resolve_status(kernel_item: Any) -> Dict[str, Any]:
        ref = getattr(kernel_item, "ref", str(kernel_item))
        title = getattr(kernel_item, "title", ref)
        status_val = "UNKNOWN"
        failure_msg = None
        try:
            st = api.kernels_status(ref)
            status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
            failure_msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", None)
        except Exception:
            pass
        return {
            "ref": ref,
            "title": title,
            "status": status_val,
            "failure_message": failure_msg
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        kernels_data = list(executor.map(_resolve_status, raw_kernels))

    return kernels_data


def get_registered_models() -> List[str]:
    """Loads model manifold names registered in unified_models_v2.yaml."""
    yaml_path = Path(__file__).resolve().parent.parent / "unified_models_v2.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return [k for k in data.keys() if k != "_registry_metadata"]
        except Exception:
            pass
    return []


def match_model_from_slug(slug: Optional[str]) -> Optional[str]:
    """Infers registered model manifold key from kernel slug."""
    if not slug:
        return None
    models = get_registered_models()
    clean_slug = slug.lower().replace("-", "").replace("_", "")
    for m in models:
        clean_m = m.replace("_", "").lower()
        if clean_m in clean_slug:
            return m
    if "forex" in clean_slug:
        return "forex_predictor"
    return None


def parse_kernel_logs(raw_logs: Any) -> str:
    """
    Parses raw API log response (JSON array of stream objects or plain string) into text lines.
    """
    if not raw_logs:
        return ""

    if isinstance(raw_logs, str):
        try:
            entries = json.loads(raw_logs)
            if isinstance(entries, list):
                chunks = []
                for item in entries:
                    if isinstance(item, dict) and "data" in item:
                        chunks.append(str(item["data"]))
                return "".join(chunks)
            if isinstance(entries, dict) and "data" in entries:
                return str(entries["data"])
        except (ValueError, TypeError):
            return raw_logs
    elif isinstance(raw_logs, list):
        chunks = []
        for item in raw_logs:
            if isinstance(item, dict) and "data" in item:
                chunks.append(str(item["data"]))
            else:
                chunks.append(str(item))
        return "".join(chunks)

    return str(raw_logs)


def stream_kernel_logs(
    api: Any,
    kernel_slug: str,
    poll_interval: int = 5,
    auto_pull: bool = False,
    username: Optional[str] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Streams live kernel logs and status telemetry to terminal.
    If auto_pull is True, pulls checkpoints and weights to local LemGendaryModels per completed epoch.
    If auto_pull is False, operates in monitor-only mode with zero checkpoint pulling.
    """
    print("\n" + "=" * 76)
    print(f"  [TELEMETRY] MONITORING KERNEL: {kernel_slug}")
    if auto_pull and model_name:
        print(f"  [AUTO-PULL ACTIVE] Checkpoints will sync to LemGendaryModels/{model_name}")
    else:
        print("  [MONITOR ONLY] Checkpoint auto-pulling is disabled.")
    print("=" * 76)

    try:
        st = api.kernels_status(kernel_slug)
        current_status = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
    except Exception as exc:
        print(f"[ERROR] Could not fetch status for {kernel_slug}: {exc}")
        return

    print(f"Current Status: {current_status}")
    print("Press Ctrl+C at any time to return to menu.\n")

    printed_lines_count = 0

    if current_status in ["COMPLETE", "ERROR", "CANCELACK"]:
        print(f"[INFO] Kernel is in terminal state: {current_status}")
        try:
            raw_logs = api.kernels_logs(kernel_slug)
            log_text = parse_kernel_logs(raw_logs)
            if log_text.strip():
                lines = log_text.splitlines()
                print(f"--- Persisted Execution Logs ({len(lines)} lines) ---")
                if len(lines) > 100:
                    print(f"[NOTE] Displaying trailing 100 lines of {len(lines)} total lines:")
                    for line in lines[-100:]:
                        print(line)
                else:
                    for line in lines:
                        print(line)
                print("--- End of Logs ---")
            else:
                print("[INFO] No log output available for this kernel run.")
        except Exception as log_exc:
            print(f"[WARN] Failed to fetch execution logs: {log_exc}")

        try:
            failure_msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", None)
            if failure_msg:
                print(f"\n[FAILURE DETAILS] {failure_msg}")
        except Exception:
            pass

        if auto_pull and model_name and current_status == "COMPLETE":
            print(f"\n[SYNC] Performing artifact pull for completed run >> {model_name} <<...")
            try:
                pull_kaggle_artifacts(model_name, username=username)
            except Exception as pull_err:
                print(f"[WARN] Artifact pull notice: {pull_err}")
        return

    # Kernel is running or queued: live polling loop
    print(f"[LIVE] Streaming telemetry every {poll_interval}s...")
    last_status = current_status

    try:
        while True:
            # Poll status
            try:
                st = api.kernels_status(kernel_slug)
                status = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
                if status != last_status:
                    print(f"\n[{time.strftime('%H:%M:%S')}] [SIGNAL] Cloud Status: {status}")
                    last_status = status
            except Exception:
                status = last_status

            # Poll logs
            try:
                raw_logs = api.kernels_logs(kernel_slug)
                log_text = parse_kernel_logs(raw_logs)
                all_lines = log_text.splitlines()
                if len(all_lines) > printed_lines_count:
                    new_lines = all_lines[printed_lines_count:]
                    for line in new_lines:
                        print(line)
                        # Check for epoch completion or checkpoint save if auto-pull is enabled
                        if auto_pull and model_name:
                            epoch_signal = re.search(
                                r"epoch\s+\d+\s+complete|saving.*checkpoint|model version.*committed|\[sota\]",
                                line,
                                re.IGNORECASE
                            )
                            if epoch_signal:
                                print(f"\n[{time.strftime('%H:%M:%S')}] [AUTO-PULL] New checkpoint detected. Syncing >> {model_name} <<...")
                                try:
                                    pull_kaggle_artifacts(model_name, username=username)
                                except Exception as pull_err:
                                    print(f"[WARN] Auto-pull notice: {pull_err}")
                    printed_lines_count = len(all_lines)
            except Exception:
                pass

            if status in ["COMPLETE", "ERROR", "CANCELACK"]:
                print(f"\n[{time.strftime('%H:%M:%S')}] [TERMINAL] Cloud job finished with status: {status}")
                try:
                    failure_msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", None)
                    if failure_msg:
                        print(f"[FAILURE DETAILS] {failure_msg}")
                except Exception:
                    pass

                if auto_pull and model_name and status == "COMPLETE":
                    print(f"\n[SYNC] Training complete. Performing final artifact pull for >> {model_name} <<...")
                    try:
                        pull_kaggle_artifacts(model_name, username=username)
                    except Exception as pull_err:
                        print(f"[WARN] Final artifact pull notice: {pull_err}")
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Detached from telemetry stream for {kernel_slug}. Process continues on Kaggle.")


def interactive_kernel_monitor(api: Any, username: str) -> None:
    """
    Submenu for selecting which kernel to monitor (Monitor Only: no pulling).
    """
    while True:
        print("\n" + "=" * 76)
        print(f"  [MONITOR ONLY] ACTIVE & RECENT KERNELS FOR: {username}")
        print("=" * 76)
        print("Querying Kaggle for active and recent kernels...")

        kernels = fetch_user_kernels(api, username, limit=20)

        if kernels:
            for idx, k in enumerate(kernels, start=1):
                status_tag = f"[{k['status']}]"
                if k['status'] in ["RUNNING", "QUEUED"]:
                    status_tag = f"[ACTIVE: {k['status']}]"
                slug_name = k['ref'].split('/')[-1]
                print(f"  {idx}. {slug_name:<40} {status_tag}")
        else:
            print("  No recent kernels found on Kaggle for this user.")

        custom_opt_idx = len(kernels) + 1
        print(f"\n  {custom_opt_idx}. [CUSTOM] Enter custom kernel slug / model name")
        print("  R. [REFRESH] Re-query kernel statuses")
        print("  B. [BACK] Return to User Selection")
        print("")

        choice = input(f"Select kernel to monitor (1-{custom_opt_idx}, R, B): ").strip()

        if choice.upper() in ["B", "Q"]:
            break
        if choice.upper() == "R":
            continue

        target_slug: Optional[str] = None

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(kernels):
                target_slug = kernels[val - 1]["ref"]
            elif val == custom_opt_idx:
                raw_input = input("\nEnter kernel slug (e.g. username/kernel-name or kernel-name): ").strip()
                if raw_input:
                    if "/" in raw_input:
                        target_slug = raw_input
                    else:
                        target_slug = f"{username}/{raw_input}"

        if target_slug:
            # Monitor Only mode: auto_pull=False
            stream_kernel_logs(api, target_slug, auto_pull=False, username=username)
            input("\nPress Enter to return to kernels list...")


def interactive_train_launcher(api: Any, username: str) -> None:
    """
    Submenu for launching training on Kaggle and streaming logs with per-epoch checkpoint pulling.
    """
    while True:
        print("\n" + "=" * 76)
        print(f"  [TRAIN ON KAGGLE] SELECT NOTEBOOK FOR USER: {username}")
        print("=" * 76)
        print("Querying user's Kaggle notebooks...")

        kernels = fetch_user_kernels(api, username, limit=20)

        if kernels:
            for idx, k in enumerate(kernels, start=1):
                status_tag = f"[{k['status']}]"
                if k['status'] in ["RUNNING", "QUEUED"]:
                    status_tag = f"[ACTIVE: {k['status']}]"
                slug_name = k['ref'].split('/')[-1]
                print(f"  {idx}. {slug_name:<40} {status_tag}")
        else:
            print("  No existing notebooks found on Kaggle.")

        new_opt_idx = len(kernels) + 1
        print(f"\n  {new_opt_idx}. [DEPLOY NEW MODEL] Deploy and launch a registered model from local suite")
        print("  B. [BACK] Return to User Selection")
        print("")

        choice = input(f"Select notebook to run (1-{new_opt_idx}, B): ").strip()

        if choice.upper() in ["B", "Q"]:
            break

        selected_slug: Optional[str] = None
        target_model: Optional[str] = None

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(kernels):
                selected_slug = kernels[val - 1]["ref"]
                target_model = match_model_from_slug(selected_slug)
                if not target_model:
                    prompt_m = input("Enter target model key for checkpoint saving (e.g. mirnet_exposure): ").strip()
                    if prompt_m:
                        target_model = prompt_m
                
                print(f"\n[LAUNCH] Pulling notebook definition for '{selected_slug}'...")
                with tempfile.TemporaryDirectory() as td:
                    try:
                        api.kernels_pull(selected_slug, path=td, metadata=True)
                        meta_file = Path(td) / "kernel-metadata.json"
                        meta_data = {}
                        if meta_file.exists():
                            try:
                                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                                meta_data["enable_gpu"] = "true"
                                meta_data["enable_internet"] = "true"
                                if "is_private" not in meta_data:
                                    meta_data["is_private"] = "true"
                                raw_shape = str(meta_data.get("machine_shape", "")).strip()
                                # Normalize legacy or non-specific shapes (like 'Gpu' or empty) to explicit accelerator
                                if not raw_shape or raw_shape.lower() in ["gpu", "none", "cpu"]:
                                    meta_data["machine_shape"] = "NvidiaTeslaT4"
                                meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
                            except Exception as meta_err:
                                print(f"[REMEDY] Metadata update notice: {meta_err}")

                        # Harden notebook sentinel cells against premature sys.exit(1) on transient container initialization
                        try:
                            for nb_path in Path(td).glob("*.ipynb"):
                                nb_text = nb_path.read_text(encoding="utf-8")
                                nb_json = json.loads(nb_text)
                                modified_nb = False
                                for cell in nb_json.get("cells", []):
                                    if cell.get("cell_type") == "code":
                                        src = "".join(cell.get("source", []))
                                        if "NO GPU DETECTED! Training aborted to preserve quota." in src and "sys.exit(1)" in src:
                                            # Replace hard abort with non-fatal warning matching notebook_generator standard
                                            src_mod = src.replace(
                                                "    print('[ERROR] [CRITICAL] NO GPU DETECTED! Training aborted to preserve quota.')\n    sys.exit(1)",
                                                "    print('[WARNING] NO GPU DETECTED!')\n    print('   -> Continuing in fallback execution mode...')"
                                            )
                                            cell["source"] = src_mod
                                            modified_nb = True
                                if modified_nb:
                                    nb_path.write_text(json.dumps(nb_json, indent=2), encoding="utf-8")
                        except Exception as nb_patch_err:
                            print(f"[REMEDY] Notebook sentinel sync notice: {nb_patch_err}")

                        chosen_acc = meta_data.get("machine_shape", "NvidiaTeslaT4") if meta_data else "NvidiaTeslaT4"
                        if str(chosen_acc).lower() in ["gpu", "none", "cpu"]:
                            chosen_acc = "NvidiaTeslaT4"
                        print(f"[LAUNCH] Pushing kernel bundle to trigger execution on Kaggle GPU ({chosen_acc})...")
                        api.kernels_push(td, acc=chosen_acc)
                        print(f"[SUCCESS] Kernel '{selected_slug}' pushed and queued on Kaggle GPU!")
                    except Exception as push_err:
                        print(f"[ERROR] Failed to launch kernel: {push_err}")
                        input("\nPress Enter to continue...")
                        continue

            elif val == new_opt_idx:
                models = get_registered_models()
                print("\n--- SELECT LOCAL MODEL MANIFOLD TO DEPLOY ---")
                for m_idx, m_name in enumerate(models, start=1):
                    print(f"  {m_idx}. {m_name}")
                m_choice = input(f"Select model (1-{len(models)}): ").strip()
                if m_choice.isdigit() and 1 <= int(m_choice) <= len(models):
                    target_model = models[int(m_choice) - 1]
                    kernel_dir = create_cloud_kernel_bundle(target_model, username, gpu="T4")
                    selected_slug = get_kernel_slug(target_model, username)
                    print(f"[LAUNCH] Pushing bundle for '{target_model}' to Kaggle GPU (Dual T4: {selected_slug})...")
                    try:
                        api.kernels_push(str(kernel_dir), acc="NvidiaTeslaT4")
                        print(f"[SUCCESS] Kernel '{selected_slug}' pushed and queued on Kaggle GPU!")
                    except Exception as push_err:
                        print(f"[ERROR] Failed to push kernel: {push_err}")
                        input("\nPress Enter to continue...")
                        continue
                else:
                    continue

        if selected_slug:
            # Train mode: auto_pull=True
            stream_kernel_logs(
                api,
                selected_slug,
                poll_interval=5,
                auto_pull=True,
                username=username,
                model_name=target_model
            )
            input("\nPress Enter to return to notebooks list...")


def prompt_user_account_selection(users_file_override: Optional[str] = None) -> Optional[Tuple[str, str, Path]]:
    """
    Presents the user account selection menu from .kaggle_users with new user addition.
    """
    explicit_path = Path(users_file_override) if users_file_override else None
    accounts, users_file = load_kaggle_users(explicit_path)

    print("\n" + "=" * 76)
    print("  LEMGENDARY KAGGLE CLOUD ENGINE (ACCOUNT SELECTOR)")
    print("=" * 76)
    print(f"Registry: {users_file}\n")

    if accounts:
        print("Select Kaggle User Account:")
        for idx, acc in enumerate(accounts, start=1):
            token_masked = acc['token'][:8] + "..." if len(acc['token']) > 8 else "***"
            print(f"  {idx}. {acc['username']:<20} (Token: {token_masked})")
    else:
        print("No saved Kaggle accounts found in .kaggle_users.")

    new_user_idx = len(accounts) + 1
    print(f"\n  {new_user_idx}. [NEW USER] Enter username and API token")
    print("  Q. [CANCEL] Return to Main Menu\n")

    choice = input(f"Selection (1-{new_user_idx}, Q): ").strip()

    if choice.upper() in ["Q", "QUIT", "EXIT", "B"]:
        return None

    selected_user: Optional[str] = None
    selected_token: Optional[str] = None

    if choice.isdigit():
        val = int(choice)
        if 1 <= val <= len(accounts):
            selected_user = accounts[val - 1]["username"]
            selected_token = accounts[val - 1]["token"]
        elif val == new_user_idx:
            print("\n--- ENTER NEW KAGGLE USER CREDENTIALS ---")
            inp_user = input("Kaggle Username: ").strip()
            inp_token = input("Kaggle API Token (KGAT_... or key): ").strip()

            if inp_user and inp_token:
                selected_user = inp_user
                selected_token = inp_token
                save_prompt = input("Save this new user to .kaggle_users? (y/n) [Default: y]: ").strip().lower()
                if save_prompt in ["", "y", "yes"]:
                    if save_kaggle_user(users_file, inp_user, inp_token):
                        print(f"[SUCCESS] User '{inp_user}' saved to {users_file}.")
            else:
                print("[ERROR] Username and token cannot be empty.")
                time.sleep(1)
                return None

    if selected_user and selected_token:
        return (selected_user, selected_token, users_file)
    return None


def run_interactive_train(users_file_override: Optional[str] = None) -> None:
    """
    Option 1: Train on Kaggle (Select notebook, launch, stream logs, auto-pull checkpoints).
    """
    while True:
        res = prompt_user_account_selection(users_file_override)
        if not res:
            break
        username, token, _ = res
        print(f"\n[AUTH] Authenticating Kaggle API for user: {username}...")
        api = authenticate_kaggle_user(username, token)
        if not api:
            input("\nPress Enter to return to account selection...")
            continue
        print(f"[OK] Authenticated successfully as '{username}'!")
        interactive_train_launcher(api, username)


def run_interactive_monitor(users_file_override: Optional[str] = None) -> None:
    """
    Option 2: Monitor Only (Stream live logs from active training, no pulling).
    """
    while True:
        res = prompt_user_account_selection(users_file_override)
        if not res:
            break
        username, token, _ = res
        print(f"\n[AUTH] Authenticating Kaggle API for user: {username}...")
        api = authenticate_kaggle_user(username, token)
        if not api:
            input("\nPress Enter to return to account selection...")
            continue
        print(f"[OK] Authenticated successfully as '{username}'!")
        interactive_kernel_monitor(api, username)


def main():
    """
    Entry point for CLI execution.
    """
    import argparse
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Cloud Training & Monitor")
    parser.add_argument("--action", type=str, default="monitor", choices=["train", "monitor", "stream"])
    parser.add_argument("--users_file", type=str, default=None, help="Path to .kaggle_users file")
    parser.add_argument("--user", type=str, default=None, help="Target Kaggle username")
    parser.add_argument("--token", type=str, default=None, help="Kaggle API token for direct auth")
    parser.add_argument("--kernel", type=str, default=None, help="Specific kernel slug to stream")
    parser.add_argument("--model", type=str, default=None, help="Associated model manifold name")
    parser.add_argument("--auto_pull", action="store_true", help="Enable per-epoch checkpoint auto-pull")
    args = parser.parse_args()

    if args.user and args.token and args.kernel:
        api = authenticate_kaggle_user(args.user, args.token)
        if api:
            slug = args.kernel if "/" in args.kernel else f"{args.user}/{args.kernel}"
            stream_kernel_logs(
                api,
                slug,
                auto_pull=args.auto_pull,
                username=args.user,
                model_name=args.model
            )
        else:
            sys.exit(1)
    elif args.action == "train":
        run_interactive_train(args.users_file)
    else:
        run_interactive_monitor(args.users_file)


if __name__ == "__main__":
    main()
