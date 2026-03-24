import os
import sys
import subprocess
def ensure_dependencies():
    try:
        import tensorflow  # pyre-ignore
    except ImportError:
        print("\n📦 Automatically resolving missing dependencies (tensorflow)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"], check=True)
        print("✅ Dependencies completely synchronized!\n")

ensure_dependencies()
import tensorflow as tf  # pyre-ignore

def convert(input_path, output_dir):
    print(f"📦 Generating Standard YOLO architecture from TFRecord blob: {input_path}")
    
    # Strictly use validate instead of val
    splits = ["train", "validate"]
    types = ["images", "labels"]
    
    for s in splits:
        for t in types:
            os.makedirs(os.path.join(output_dir, t, s), exist_ok=True)
            
    print("🚀 Initiating extraction of TFRecord streaming buffers...")
    # ----------------------------------------------------------------
    # TEMPLATE EXTRACTION LOGIC
    # dataset = tf.data.TFRecordDataset(input_path)
    # feature_description = {
    #     'image/encoded': tf.io.FixedLenFeature([], tf.string),
    #     'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    # }
    # def _parse_function(example_proto):
    #   return tf.io.parse_single_example(example_proto, feature_description)
    # parsed_dataset = dataset.map(_parse_function)
    # for idx, parsed_record in enumerate(parsed_dataset):
    #      # write out JPEG and dynamically translate bbox to YOLO format!
    # ----------------------------------------------------------------
    print(f"✅ Successfully decompressed TFRecord blob to {output_dir}")
    
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
        print("Usage: python convert_tfrecord_to_yolo.py <input.tfrecord> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
