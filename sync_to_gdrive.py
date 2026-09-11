#!/usr/bin/env python3
"""
LemGendary Model Synchronizer -> Google Drive
============================================
Syncs model checkpoints, metrics, and exported binaries from LemGendaryModels
directly to Google Drive root folder (142G7B9ONfUkXAhVkPeN4NeJ3YXU0UmJX).

Usage:
    python sync_to_gdrive.py --model <model_key>
    python sync_to_gdrive.py --all
    python sync_to_gdrive.py --model <model_key> --dry-run
"""

from training.gdrive_cloud_manager import main

if __name__ == "__main__":
    main()
