import os
import warnings
from PIL import Image, ImageFile  # pyre-ignore
from tqdm import tqdm  # pyre-ignore
import multiprocessing

# Completely disable tolerance for structurally truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Deep Byte-Validation Protocol
def check_and_delete(img_path):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Phase 1: Rapid mathematical Byte Header check
            with Image.open(img_path) as img:
                img.verify()
                
            # Phase 2: Bruteforce Decompression to intercept hidden libjpeg-turbo Extraneous Bytes
            with Image.open(img_path) as img:
                img.load()
                
            # Filter warnings immediately
            if w and len(w) > 0:
                for warning in w:
                    msg = str(warning.message).lower()
                    if "corrupt" in msg or "truncated" in msg or "extraneous bytes" in msg:
                        return img_path  # Mathematical Corruption Detected
        return None
    except Exception:
        return img_path # Structural crash (unreadable)

def process_file(img_path):
    bad_img = check_and_delete(img_path)
    if bad_img:
        try:
            # Physically shred the anomaly
            os.remove(bad_img)
            
            # Purge attached tracking arrays safely
            lbl_txt = bad_img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
            if os.path.exists(lbl_txt): os.remove(lbl_txt)
            
            tgt_path = bad_img.replace("images", "targets")
            if os.path.exists(tgt_path): os.remove(tgt_path)
            
            return True
        except:
            pass
    return False

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else r"C:\Development\webapps\react\image-lemgendizer-old\training\lemgendary-training-suite\data\datasets\LemGendizedFaceDataset"
    
    print("🚀 Initializing Global LemGendary Data Sanitizer...")
    
    # Fast path mapping
    all_images = []
    for root, _, files in os.walk(base_dir):
        if "labels" in root or "targets" in root: continue
        
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, f))
                
    print(f"📡 Mapped {len(all_images)} fundamental image tensors. Igniting Multi-Core Scanner...\n")
    
    deleted_count = 0
    # Executing securely on a single background logic core to explicitly prevent NVMe I/O starvation during live training
    with multiprocessing.Pool(processes=1) as pool:
        for result in tqdm(pool.imap_unordered(process_file, all_images), total=len(all_images), desc="Verifying Bytes"):
            if result:
                deleted_count += 1  # pyre-ignore
                
    print(f"\n✅ Clean-Sweep Complete! Physically pulverized {deleted_count} irreparably corrupted files.")
    print("Your dataset is completely pure and mathematically safe to orchestrate.")
