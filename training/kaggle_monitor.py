"""Kaggle cloud training launcher and live kernel monitor (command-line entry point).

Selects a Kaggle account, then opens the train launcher or the monitor menu
(``training.kaggle_launcher``); kernel log streaming and checkpoint pulls live in
``training.kaggle_stream``, the account registry in ``training.kaggle_accounts``. Run as
``python -m training.kaggle_monitor`` from the suite root.
"""

import argparse
import sys
import time
from typing import Any

from training.cloud.kaggle_client import KaggleAuthError, authenticated_kaggle_api
from training.kaggle_accounts import vault_accounts
from training.kaggle_launcher import interactive_kernel_monitor, interactive_train_launcher
from training.kaggle_stream import AutoPull, stream_kernel_logs


def _prompt_session_account() -> tuple[str, str] | None:
    """Ask for a username and token used for this session only (never saved)."""
    print("\n--- KAGGLE CREDENTIALS FOR THIS SESSION (not saved) ---")
    inp_user = input("Kaggle Username: ").strip()
    inp_token = input("Kaggle API Token (KGAT_...): ").strip()
    if inp_user and inp_token:
        return inp_user, inp_token
    print("[ERROR] Username and token cannot be empty.")
    time.sleep(1)
    return None


def prompt_user_account_selection() -> tuple[str, str] | None:
    """Account menu: the vault's Kaggle accounts, or credentials for this session only.

    Accounts are added to the vault with the environment manager; nothing is written here.
    """
    accounts = vault_accounts()

    print("\n" + "=" * 76)
    print("  LEMGENDARY KAGGLE CLOUD ENGINE (ACCOUNT SELECTOR)")
    print("=" * 76)

    if accounts:
        print("Select Kaggle User Account (secrets vault):")
        for idx, acc in enumerate(accounts, start=1):
            print(f"  {idx}. {acc['username']}")
    else:
        print("No Kaggle accounts in the secrets vault (add one with the environment manager).")

    session_idx = len(accounts) + 1
    print(f"\n  {session_idx}. [SESSION] Enter a username and API token (not saved)")
    print("  Q. [CANCEL] Return to Main Menu\n")

    choice = input(f"Selection (1-{session_idx}, Q): ").strip()
    if choice.upper() in ["Q", "QUIT", "EXIT", "B"] or not choice.isdigit():
        return None
    val = int(choice)
    if 1 <= val <= len(accounts):
        return accounts[val - 1]["username"], accounts[val - 1]["token"]
    if val == session_idx:
        return _prompt_session_account()
    return None


def _authenticate(username: str, token: str) -> Any | None:
    """Authenticated Kaggle API for the selected account, or None after printing why."""
    try:
        return authenticated_kaggle_api(username, token)
    except KaggleAuthError as exc:
        print(f"\n[ERROR] {exc}")
        print("[REMEDY] Verify username and API token validity on kaggle.com/settings/api.")
        return None


def run_interactive_train() -> None:
    """
    Option 1: Train on Kaggle (Select notebook, launch, stream logs, auto-pull checkpoints).
    """
    while True:
        res = prompt_user_account_selection()
        if not res:
            break
        username, token = res
        print(f"\n[AUTH] Authenticating Kaggle API for user: {username}...")
        api = _authenticate(username, token)
        if not api:
            input("\nPress Enter to return to account selection...")
            continue
        print(f"[OK] Authenticated successfully as '{username}'!")
        interactive_train_launcher(api, username)


def run_interactive_monitor() -> None:
    """
    Option 2: Monitor Only (Stream live logs from active training, no pulling).
    """
    while True:
        res = prompt_user_account_selection()
        if not res:
            break
        username, token = res
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
        run_interactive_train()
    else:
        run_interactive_monitor()


if __name__ == "__main__":
    main()
