import os
import json
import numpy as np
import cv2
from pathlib import Path

class ImageTransformer:
    def __init__(self, coords_file):
        """Initialize with path to JSON coordinates file"""
        with open(coords_file) as f:
            self.coords_data = json.load(f)
    
    def transform_image(
        self,
        image_name,
        input_dir,
        output_dir,
        desired_left_eye_pos,  # tuple of (x%, y%) in 0-1 range
        desired_right_eye_pos, # tuple of (x%, y%) in 0-1 range
        output_size=(4032, 3024)
    ):
        """
        Transform a single image to align eyes with desired positions
        
        Args:
            image_name: Name of image (without extension)
            input_dir: Directory containing source images
            output_dir: Directory to save transformed images
            desired_left_eye_pos: Target position for left eye (x%, y%)
            desired_right_eye_pos: Target position for right eye (x%, y%)
            output_size: Output dimensions (width, height)
        """
        # Load image
        img_path = Path(input_dir) / f"{image_name}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image {img_path}")

        # Get current eye positions from JSON
        img_data = self.coords_data.get(image_name)
        if not img_data or not img_data.get("left_eye") or not img_data.get("right_eye"):
            print(f"Skipping transforming {image_name} - no eye coordinates")
            return None

        # Convert percentage positions to pixel coordinates
        width, height = output_size
        def percent_to_pixels(percent_pos):
            return (int(percent_pos[0] * width), (int(percent_pos[1] * height)))

        # Prepare source and destination points
        src_left = img_data["left_eye"]
        src_right = img_data["right_eye"]
        dst_left = percent_to_pixels(desired_left_eye_pos)
        dst_right = percent_to_pixels(desired_right_eye_pos)

        # Calculate similarity transform (rotation, scale, translation)
        src_pts = np.float32([src_left, src_right])
        dst_pts = np.float32([dst_left, dst_right])
        
        transform_mat = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

        # Apply transformation
        transformed_img = cv2.warpAffine(
            img,
            transform_mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Save result
        output_path = Path(output_dir) / f"{image_name}.png"
        cv2.imwrite(str(output_path), transformed_img)
        
        return output_path

def transform_images(imgs, src_dir, dest_dir, coords_file):
    transformer = ImageTransformer(coords_file)

    for img in imgs:
        transformer.transform_image(img, src_dir, dest_dir, (0.345, 0.521), (0.438, 0.521), (4032, 3024))
