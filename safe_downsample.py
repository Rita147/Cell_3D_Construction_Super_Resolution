# safe_downsample.py

from PIL import Image, UnidentifiedImageError
from pathlib import Path
import os
import gc
import cv2

SCALE_FACTOR = 0.5
BATCH_SIZE = 5

INPUT_ROOT = Path("data")
OUTPUT_ROOT = Path("downsampled_data")
SPLITS = ["train", "val", "test"]
MODALITIES = ["images", "masks", "distance_maps", "label_masks", "vague_masks"]

def get_resize_method(modality):
    return Image.NEAREST if modality != "images" else Image.BICUBIC

def downsample_image_safe(img_path, out_path, modality):
    try:
        with Image.open(img_path) as img:
            new_size = (int(img.width * SCALE_FACTOR), int(img.height * SCALE_FACTOR))
            resized = img.resize(new_size, resample=get_resize_method(modality))
            resized.save(out_path)
            return True
    except (UnidentifiedImageError, OSError):
        try:
            img_cv = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                return False
            new_size = (int(img_cv.shape[1] * SCALE_FACTOR), int(img_cv.shape[0] * SCALE_FACTOR))
            interp = cv2.INTER_NEAREST if modality != "images" else cv2.INTER_CUBIC
            resized_cv = cv2.resize(img_cv, new_size, interpolation=interp)
            cv2.imwrite(str(out_path), resized_cv)
            return True
        except Exception as e:
            print(f"❌ OpenCV fallback failed for {img_path.name}: {e}")
            return False

def downsample_batchwise(input_folder, output_folder, modality):
    files = sorted([f for f in input_folder.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])

    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]

        for img_path in batch:
            out_path = output_folder / img_path.name
            success = downsample_image_safe(img_path, out_path, modality)
            if success:
                print(f"✅ {modality}/{img_path.name}")
            else:
                print(f"⚠️  Skipped {modality}/{img_path.name} (corrupt or unreadable)")

        gc.collect()

for split in SPLITS:
    for modality in MODALITIES:
        input_path = INPUT_ROOT / split / modality
        output_path = OUTPUT_ROOT / split / modality
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📦 Processing {split}/{modality}...")
        downsample_batchwise(input_path, output_path, modality)

print("🎉 All done, downsampling completed.")
