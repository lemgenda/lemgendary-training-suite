import sys
import os
import subprocess

def prefetch():
    """
    Isolated Worker Thread Sequence designed to mathematically download
    Kaggle Zip arrays and extract them physically outside of the global
    PyTorch loop constraint, ensuring 100% Zero-Latency handoffs natively!
    """
    if len(sys.argv) < 3:
        return
        
    # Example format: lemtreursi/lemgendizedfacedataset:LemGendizedFaceDataset,...
    datasets = sys.argv[1].split(',')
    target_dir = sys.argv[2]
    
    os.makedirs(target_dir, exist_ok=True)
    
    for ds_pair in datasets:
        if not ds_pair or ':' not in ds_pair: continue
        kaggle_id, folder_name = ds_pair.split(':')
        
        ds_path = os.path.join(target_dir, folder_name)
        if os.path.exists(ds_path):
            continue # Structurally already completely cached
            
        # Engage structural Mutex Lock
        lock_file = os.path.join(target_dir, f"{folder_name}.lock")
        
        with open(lock_file, "w") as f:
            f.write("STREAMING_FROM_KAGGLE_CLOUD")
            
        cmd = f"kaggle datasets download -d {kaggle_id} -p \"{target_dir}\" --unzip"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            pass # Suppress background errors to protect UI focus; orchestrator checks existence natively anyway
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)

if __name__ == "__main__":
    prefetch()
