# downsample_superres_flat.py

import os
from pathlib import Path
import cv2
from tqdm import tqdm

# ========== CONFIG ==========
SCALE = 0.5
INPUT_ROOT = Path("data")
OUTPUT_ROOT = Path("downsampled_data")
FLAT_OUTPUT_ROOT = Path("downsampled_all")
SPLITS = ["train", "val", "test"]
MODALITY = "images"
INTERPOLATION = cv2.INTER_CUBIC  # Use smooth interpolation for images

# ========== UTILS ==========
def get_organ_from_filename(filename):
    return "_".join(filename.name.split("_")[:2])  # e.g., human_kidney_003.png → human_kidney

def downsample_image(img_path, save_path):
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Skipped unreadable: {img_path.name}")
            return False

        new_size = (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE))
        resized = cv2.resize(img, new_size, interpolation=INTERPOLATION)
        cv2.imwrite(str(save_path), resized)
        return True
    except Exception as e:
        print(f"❌ Error processing {img_path.name}: {e}")
        return False

# ========== MAIN ==========
for split in SPLITS:
    input_dir = INPUT_ROOT / split / MODALITY
    if not input_dir.exists():
        continue

    image_paths = sorted(input_dir.glob("*.png"))
    print(f"\n📂 Processing {split} — {len(image_paths)} images")

    output_split_dir = OUTPUT_ROOT / split / MODALITY
    output_split_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths):
        organ = get_organ_from_filename(img_path)

        # Save to split structure
        save_path_struct = output_split_dir / img_path.name

        # Save to organ folder
        organ_output_dir = FLAT_OUTPUT_ROOT / organ / MODALITY
        organ_output_dir.mkdir(parents=True, exist_ok=True)
        save_path_flat = organ_output_dir / img_path.name

        success = downsample_image(img_path, save_path_struct)
        if success:
            # Avoid re-downsampling — copy saved file
            img = cv2.imread(str(save_path_struct), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(save_path_flat), img)

    print(f"✅ Finished {split}")

print("\n🎉 ALL DONE. Downsampled by organ + split.")
