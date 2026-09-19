"""LemGendary Model Synchronizer -> Google Drive.

Syncs model checkpoints, metrics, and exported binaries from LemGendaryModels
directly to Google Drive root folder.
"""

from training.gdrive_cloud_manager import main

if __name__ == "__main__":
    main()
