"""Kaggle cloud training launcher and live kernel monitor (command-line entry point).

Selects a Kaggle account, then opens the train launcher or the monitor menu
(``training.kaggle_launcher``); kernel log streaming and checkpoint pulls live in
``training.kaggle_stream``, the account registry in ``training.kaggle_accounts``. Run as
``python -m training.kaggle_monitor`` from the suite root.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from training.cloud.kaggle_client import KaggleAuthError, authenticated_kaggle_api
from training.kaggle_accounts import load_kaggle_users, save_kaggle_user
from training.kaggle_launcher import interactive_kernel_monitor, interactive_train_launcher
from training.kaggle_stream import AutoPull, stream_kernel_logs


def prompt_user_account_selection(
    users_file_override: str | None = None,
) -> tuple[str, str, Path] | None:
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
            token_masked = acc["token"][:8] + "..." if len(acc["token"]) > 8 else "***"
            print(f"  {idx}. {acc['username']:<20} (Token: {token_masked})")
    else:
        print("No saved Kaggle accounts found in .kaggle_users.")

    new_user_idx = len(accounts) + 1
    print(f"\n  {new_user_idx}. [NEW USER] Enter username and API token")
    print("  Q. [CANCEL] Return to Main Menu\n")

    choice = input(f"Selection (1-{new_user_idx}, Q): ").strip()

    if choice.upper() in ["Q", "QUIT", "EXIT", "B"]:
        return None

    selected_user: str | None = None
    selected_token: str | None = None

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
                save_prompt = (
                    input("Save this new user to .kaggle_users? (y/n) [Default: y]: ")
                    .strip()
                    .lower()
                )
                if save_prompt in ["", "y", "yes"] and save_kaggle_user(
                    users_file, inp_user, inp_token
                ):
                    print(f"[SUCCESS] User '{inp_user}' saved to {users_file}.")
            else:
                print("[ERROR] Username and token cannot be empty.")
                time.sleep(1)
                return None

    if selected_user and selected_token:
        return (selected_user, selected_token, users_file)
    return None


def _authenticate(username: str, token: str) -> Any | None:
    """Authenticated Kaggle API for the selected account, or None after printing why."""
    try:
        return authenticated_kaggle_api(username, token)
    except KaggleAuthError as exc:
        print(f"\n[ERROR] {exc}")
        print("[REMEDY] Verify username and API token validity on kaggle.com/settings/api.")
        return None


def run_interactive_train(users_file_override: str | None = None) -> None:
    """
    Option 1: Train on Kaggle (Select notebook, launch, stream logs, auto-pull checkpoints).
    """
    while True:
        res = prompt_user_account_selection(users_file_override)
        if not res:
            break
        username, token, _ = res
        print(f"\n[AUTH] Authenticating Kaggle API for user: {username}...")
        api = _authenticate(username, token)
        if not api:
            input("\nPress Enter to return to account selection...")
            continue
        print(f"[OK] Authenticated successfully as '{username}'!")
        interactive_train_launcher(api, username)


def run_interactive_monitor(users_file_override: str | None = None) -> None:
    """
    Option 2: Monitor Only (Stream live logs from active training, no pulling).
    """
    while True:
        res = prompt_user_account_selection(users_file_override)
        if not res:
            break
        username, token, _ = res
        print(f"\n[AUTH] Authenticating Kaggle API for user: {username}...")
        api = _authenticate(username, token)
        if not api:
            input("\nPress Enter to return to account selection...")
            continue
        print(f"[OK] Authenticated successfully as '{username}'!")
        interactive_kernel_monitor(api, username)


def main():
    """
    Entry point for CLI execution.
    """
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Cloud Training & Monitor")
    parser.add_argument(
        "--action", type=str, default="monitor", choices=["train", "monitor", "stream"]
    )
    parser.add_argument("--users_file", type=str, default=None, help="Path to .kaggle_users file")
    parser.add_argument("--user", type=str, default=None, help="Target Kaggle username")
    parser.add_argument("--token", type=str, default=None, help="Kaggle API token for direct auth")
    parser.add_argument("--kernel", type=str, default=None, help="Specific kernel slug to stream")
    parser.add_argument("--model", type=str, default=None, help="Associated model manifold name")
    parser.add_argument(
        "--auto_pull", action="store_true", help="Enable per-epoch checkpoint auto-pull"
    )
    args = parser.parse_args()

    if args.user and args.token and args.kernel:
        api = _authenticate(args.user, args.token)
        if api:
            slug = args.kernel if "/" in args.kernel else f"{args.user}/{args.kernel}"
            pull = AutoPull(args.model, args.user) if args.auto_pull and args.model else None
            stream_kernel_logs(api, slug, pull=pull)
        else:
            sys.exit(1)
    elif args.action == "train":
        run_interactive_train(args.users_file)
    else:
        run_interactive_monitor(args.users_file)


if __name__ == "__main__":
    main()
