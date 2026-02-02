import os
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import pytesseract

# Configuration
INPUT_DIR = "documents"
OUTPUT_DIR = "optimized_documents"
MAX_EDGE = 1000  # Reduced resolution for smaller file sizes
JPEG_QUALITY = 68
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

def optimize_for_ocr(img):
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize (preserve aspect ratio)
    h, w = img.shape[:2]
    scale = min(MAX_EDGE / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img

def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in SUPPORTED_EXT:
                continue

            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            img = cv2.imread(input_path)
            if img is None:
                print(f"⚠️ Skipped unreadable file: {input_path}")
                continue

            optimized = optimize_for_ocr(img)

            # Save as JPEG unless original was PNG
            out_ext = ".jpg" if ext != ".png" else ".png"
            output_path = str(Path(output_path).with_suffix(out_ext))

            if out_ext == ".jpg":
                cv2.imwrite(
                    output_path,
                    optimized,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
            else:
                cv2.imwrite(output_path, optimized)

            print(f"{rel_path} → {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_directory(INPUT_DIR, OUTPUT_DIR)
    print("\n Image optimization complete!")
