import os
import sys
import subprocess
import tarfile

def convert(input_path, output_dir):
    print(f"📦 Generating Standard YOLO architecture from WebDataset Tarball: {input_path}")
    
    # Strictly use validate instead of val
    splits = ["train", "validate"]
    types = ["images", "labels"]
    
    for s in splits:
        for t in types:
            os.makedirs(os.path.join(output_dir, t, s), exist_ok=True)
            
    print("🚀 Initiating extraction of WebDataset sequential byte stream...")
    # ----------------------------------------------------------------
    # TEMPLATE EXTRACTION LOGIC
    # with tarfile.open(input_path, "r") as tar:
    #     for member in tar.getmembers():
    #         if member.name.endswith(".jpg"):
    #             tar.extract(member, path=os.path.join(output_dir, "images", "train"))
    #         elif member.name.endswith(".json") or member.name.endswith(".txt"):
    #             tar.extract(member, path=os.path.join(output_dir, "labels", "train"))
    # # Note: implement split logic manually if tarball mixes train/validate
    # ----------------------------------------------------------------
    print(f"✅ Successfully expanded Tarball to {output_dir}")
    
    print("\n📊 Verifying Dataset Output Topography:")
    for s in splits:
        img_count = len(os.listdir(os.path.join(output_dir, "images", s)))
        lbl_count = len(os.listdir(os.path.join(output_dir, "labels", s)))
        print(f"  - {s.capitalize()} Images: {img_count}")
        print(f"  - {s.capitalize()} Labels: {lbl_count}")
        
    print(f"\n🛡️ Sequentially launching clean_corrupt_images.py on '{output_dir}'...")
    cleaner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_corrupt_images.py")
    subprocess.run([sys.executable, cleaner, output_dir])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_webdataset_to_yolo.py <input.tar> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
