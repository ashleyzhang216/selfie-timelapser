import os
import imageio
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def timelapse_images(
    image_names,
    input_dir,
    output_path,
    total_duration_sec=20.0,
    loop=0,  # 0 = infinite loop (only for GIF)
    optimize=True,
    fps=None,
    resize_to=None,
    output_format='both'  # 'both', 'gif', or 'mp4'
):
    """
    Create a timelapse from aligned images
    
    Args:
        image_names: List of image filenames (without extension)
        input_dir: Directory containing the aligned PNG images
        output_path: Where to save the output file
        total_duration_sec: Total duration for all images in seconds
        loop: Number of loops (0 = infinite, GIF only)
        optimize: Whether to optimize output for smaller size
        fps: Optional: Force specific FPS instead of calculating from duration
        resize_to: Optional: (width, height) to resize images
        output_format: 'both', 'gif', or 'mp4'
    """
    # Prepare paths
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format
    if output_format == 'auto':
        output_format = output_path.suffix.lower()[1:]  # Remove dot
        if output_format not in ['both', 'gif', 'mp4']:
            raise ValueError(f"Unsupported output format: {output_format}")
    
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
            print(f"Skipping {img_name} - no aligned version present")
    
    if not frames:
        raise ValueError("No valid images found for timelapse")
    
    # Calculate frame duration/rate
    num_images = len(frames)
    if fps:
        frame_duration = 1/fps
    else:
        frame_duration = total_duration_sec / num_images
    
    # Save in appropriate format
    if output_format == 'gif' or output_format == 'both':
        _save_as_gif(frames, output_path.with_suffix('.gif'), frame_duration, loop, optimize)
    if output_format == 'mp4' or output_format == 'both':
        _save_as_mp4(frames, output_path.with_suffix('.mp4'), frame_duration, fps, resize_to)        

def _save_as_gif(frames, output_path, frame_duration, loop, optimize):
    """Save frames as animated GIF"""
    duration_ms = frame_duration * 1000  # Convert to milliseconds
    
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=loop,
        optimize=optimize
    )
    print(f"Created GIF timelapse with {len(frames)} frames at {output_path}")
    print(f"Frame duration: {duration_ms:.1f}ms ({1000/duration_ms:.1f} FPS)")

def _save_as_mp4(frames, output_path, frame_duration, fps, resize_to):
    """Save frames as MP4 video"""
    if fps is None:
        fps = 1 / frame_duration
    
    # Convert PIL Images to numpy arrays
    frames_np = [np.array(frame) for frame in frames]
    
    # Get dimensions from first frame
    height, width = frames_np[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video.isOpened():
        raise RuntimeError("Failed to initialize video writer")
    
    # Write frames
    for frame in frames_np:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    
    video.release()
    print(f"Created MP4 timelapse with {len(frames)} frames at {output_path}")
    print(f"Output resolution: {width}x{height}, FPS: {fps:.1f}")
