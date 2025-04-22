import os
import sys
from pathlib import Path
from file_conversion import convert_heic_to_png
from gui import get_eye_coords

ROOT_DATA_PATH = "data/"
RAW_DATA_PATH = ROOT_DATA_PATH + "raw/"
CONVERTED_DATA_PATH = ROOT_DATA_PATH + "png/"

def make_dirs():
    if not os.path.exists(ROOT_DATA_PATH):
        os.makedirs(ROOT_DATA_PATH)

    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
    if not os.path.exists(CONVERTED_DATA_PATH):
        os.makedirs(CONVERTED_DATA_PATH)

if __name__ == "__main__":
    make_dirs()
    imgs = convert_heic_to_png(RAW_DATA_PATH, CONVERTED_DATA_PATH)
    if not imgs:
        print("No images found")
        sys.exit(1)

    for img in imgs:
        img_path = Path(os.path.join(CONVERTED_DATA_PATH, img + '.png'))
        coords = get_eye_coords(img_path)
        print(img)
        print(coords)
        break


