import os
import sys
from pathlib import Path
from file_conversion import convert_heic_to_png
from gui import get_eye_coords
from labeling import label_images
from transform import transform_images
from ordering import order_images
from timelapse import timelapse_images

ROOT_DATA_PATH = "data/"
RAW_DATA_PATH = ROOT_DATA_PATH + "raw/"
CONVERTED_DATA_PATH = ROOT_DATA_PATH + "png/"
EYE_DATA_PATH = ROOT_DATA_PATH + "eye_coords/"
ALIGNED_DATA_PATH = ROOT_DATA_PATH + "aligned/"
THUMBNAIL_DATA_PATH = ROOT_DATA_PATH + "thumbnails/"
OUTPUT_DATA_PATH = ROOT_DATA_PATH + "out/"

def make_dirs():
    if not os.path.exists(ROOT_DATA_PATH):
        os.makedirs(ROOT_DATA_PATH)

    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
    if not os.path.exists(CONVERTED_DATA_PATH):
        os.makedirs(CONVERTED_DATA_PATH)
    if not os.path.exists(EYE_DATA_PATH):
        os.makedirs(EYE_DATA_PATH)
    if not os.path.exists(ALIGNED_DATA_PATH):
        os.makedirs(ALIGNED_DATA_PATH)
    if not os.path.exists(THUMBNAIL_DATA_PATH):
        os.makedirs(THUMBNAIL_DATA_PATH)
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH)

if __name__ == "__main__":
    make_dirs()
    imgs = convert_heic_to_png(RAW_DATA_PATH, CONVERTED_DATA_PATH)
    if not imgs:
        print("No images found")
        sys.exit(1)
    
    imgs = imgs[:50]

    eye_coords_path = Path(os.path.join(EYE_DATA_PATH, "ashley" + '.json'))
    label_images(imgs, CONVERTED_DATA_PATH, eye_coords_path, False)
    # transform_images(imgs, CONVERTED_DATA_PATH, ALIGNED_DATA_PATH, THUMBNAIL_DATA_PATH, eye_coords_path)
    imgs = order_images(imgs, THUMBNAIL_DATA_PATH, True)

    output_path = Path(os.path.join(OUTPUT_DATA_PATH, "ashley" + '.gif'))
    timelapse_images(imgs, ALIGNED_DATA_PATH, output_path, 20.0, resize_to=(1024, 768))
