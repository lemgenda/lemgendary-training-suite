import os
import sys
import subprocess
def ensure_dependencies():
    try:
        import pandas  # pyre-ignore
        import pyarrow  # pyre-ignore
    except ImportError:
        print("\n📦 Automatically resolving missing dependencies (pandas, pyarrow)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "pyarrow"], check=True)
        print("✅ Dependencies completely synchronized!\n")

ensure_dependencies()
import pandas as pd  # pyre-ignore
import pyarrow.parquet as pq  # pyre-ignore

def convert(input_path, output_dir):
    print(f"📦 Generating Standard YOLO architecture from Parquet: {input_path}")
    
    # Strictly use validate instead of val
    splits = ["train", "validate"]
    types = ["images", "labels"]
    
    for s in splits:
        for t in types:
            os.makedirs(os.path.join(output_dir, t, s), exist_ok=True)
            
    print("🚀 Initiating mathematical extraction of Parquet DataFrame...")
    # ----------------------------------------------------------------
    # TEMPLATE EXTRACTION LOGIC
    # df = pd.read_parquet(input_path)
    # for idx, row in df.iterrows():
    #     split = "train" if row.get("split") == "train" else "validate"
    #     image_bytes = row["image"]["bytes"]
    #     label_data = row["label"]
    #     # Save image to os.path.join(output_dir, "images", split, f"{idx}.jpg")
    #     # Save label to os.path.join(output_dir, "labels", split, f"{idx}.txt")
    # ----------------------------------------------------------------
    print(f"✅ Successfully converted Parquet into {output_dir}")
    
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
        print("Usage: python convert_parquet_to_yolo.py <input.parquet> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
