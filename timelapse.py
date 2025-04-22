import os
import imageio
from pathlib import Path
from PIL import Image

def timelapse_images(
    image_names,
    input_dir,
    output_path,
    total_duration_sec,
    loop=0,  # 0 = infinite loop
    optimize=True,
    fps=None,
    resize_to=None
):
    """
    Create a timelapse GIF from aligned images
    
    Args:
        image_names: List of image filenames (without extension)
        input_dir: Directory containing the aligned PNG images
        output_path: Where to save the GIF (e.g., 'output/timelapse.gif')
        total_duration_sec: Total duration for all images in seconds
        loop: Number of loops (0 = infinite)
        optimize: Whether to optimize GIF for smaller size
        fps: Optional: Force specific FPS instead of calculating from duration
        resize_to: Optional: (width, height) to resize images
    """
    # Prepare paths
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and optionally resize images
    frames = []
    for img_name in image_names:
        img_path = input_dir / f"{img_name}.png"
        if Path(img_path).exists():
            with Image.open(img_path) as img:
                if resize_to:
                    img = img.resize(resize_to, Image.LANCZOS)
                frames.append(img.copy())
        else:
            print(f"Skipping timelapsing {img_name} - no aligned version present")
    
    # Calculate frame duration
    num_images = len(frames)
    if fps:
        duration_ms = 1000 / fps  # Convert fps to milliseconds per frame
    else:
        duration_ms = (total_duration_sec * 1000) / num_images
    
    # Save as GIF
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=loop,
        optimize=optimize
    )
    
    print(f"Created timelapse with {num_images} frames at {output_path}")
    print(f"Frame duration: {duration_ms:.1f}ms ({1000/duration_ms:.1f} FPS)")
