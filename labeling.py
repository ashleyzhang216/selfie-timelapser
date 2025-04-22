import json
from pathlib import Path
from gui import get_eye_coords
from PIL import Image
import os

def label_images(imgs, img_dir, output_json, relabel=False):
    """
    Process images with persistent JSON storage and relabel option.
    
    Args:
        imgs: List of image names
        img_dir: Directory to images
        output_json: Path to JSON storage file
        relabel: If True, allows re-labeling existing images
    """
    # Load existing data if available
    if os.path.exists(output_json):
        with open(output_json) as f:
            results = json.load(f)
    else:
        results = {}
    
    for img in imgs:
        try:
            path = Path(os.path.join(img_dir, img + '.png'))
            print(f"Labeling: {path.name}")
            
            # Check if image already has coordinates
            existing_data = results.get(img)
            if existing_data and not relabel:
                print(f"Skipping (already labeled, relabel=False)")
                continue
            
            # Get image dimensions
            with Image.open(path) as image:
                width, height = image.size
            
            # Get existing coordinates if available
            initial_coords = None
            if existing_data and relabel:
                if existing_data["left_eye"] is not None and existing_data["right_eye"] is not None:
                    initial_coords = [
                        existing_data["left_eye"],
                        existing_data["right_eye"]
                    ]

            # Get eye coordinates (pass initial positions if available)
            coords = get_eye_coords(path, initial_positions=initial_coords)

            # Prepare eye data (null if not exactly 2 eyes)
            if coords and len(coords) == 2:
                coords_sorted = sorted(coords, key=lambda c: c[0])
                eye_data = {
                    "left_eye": list(coords_sorted[0]),
                    "right_eye": list(coords_sorted[1])
                }
            else:
                eye_data = {
                    "left_eye": None,
                    "right_eye": None
                }
            
            # Update results
            results[img] = {
                "width": width,
                "height": height,
                **eye_data
            }
            
            # Save after each image (in case of crash)
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2)
            
            if coords and len(coords) == 2:
                print(f"Saved eye coords data for {path.name}")
            else:
                print(f"Saved null data for {path.name}")
            
        except Exception as e:
            print(f"Error processing {path.name}: {str(e)}")
            continue
    
    print(f"\nAll labeling results saved to {output_json}")
    return results
