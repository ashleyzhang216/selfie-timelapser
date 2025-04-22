import os
from pathlib import Path
from pillow_heif import register_heif_opener
register_heif_opener()

from PIL import Image

def convert_heic_to_png(input_dir, output_dir=None) -> list[str]:
    """
    Convert all HEIC images in input_dir to PNG format.
    If output_dir is not specified, images will be saved in the same directory.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    heic_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.heic')]
    img_names = []

    for heic_file in heic_files:
        heic_path = Path(os.path.join(input_dir, heic_file))
        png_path = Path(os.path.join(output_dir, Path(heic_file).stem + '.png'))
        img_names.append(Path(heic_file).stem)

        if not png_path.exists():
            image = Image.open(heic_path)
            image.save(png_path)
        
        assert(png_path.exists() and png_path.is_file())

    img_names.sort()
    return img_names