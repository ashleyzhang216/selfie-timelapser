import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

class DotPlacer:
    def __init__(self, image_path, initial_positions=None):
        self.root = tk.Tk()
        self.root.title("Labeling " + image_path.name)
        
        # Load image
        self.original_image = Image.open(image_path)
        self.image = self.original_image.copy()
        self.tk_image = None
        self.scale_factor = 1.0
        
        # Set up window with reasonable starting size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(800, screen_width - 100)
        window_height = min(600, screen_height - 100)
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(self.root)
        h_scroll = tk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        v_scroll = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Dot storage
        self.dots = []
        self.dot_ids = []
        self.coords = []
        
        # Bind events
        self.canvas.bind("<Button-1>", self.place_dot)
        self.root.bind("<Return>", self.finish)
        self.root.bind("<BackSpace>", self.clear_dots)
        self.root.bind("<Configure>", self.resize_image)
        
        # Initial render
        self.resize_image()

        # Use initial positions if provided
        if initial_positions and len(initial_positions) == 2:
            for pos in initial_positions:
                self.place_dot(None, pos)
        
        self.root.mainloop()
    
    def resize_image(self, event=None):
        # Get available space
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Calculate scaling while maintaining aspect ratio
        img_width, img_height = self.original_image.size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        self.scale_factor = min(width_ratio, height_ratio)
        
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        # Resize the image
        self.image = self.original_image.resize(
            (new_width, new_height), 
            Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Redraw dots at scaled positions
        self.redraw_dots()
        
        # Add instructions
        instructions = ("Place two red dots (e.g., for eyes)\n"
                       "Click to place/move dots\n"
                       "Press 'Enter' when done")
        self.canvas.create_text(
            new_width//2, 20, 
            text=instructions, 
            fill="white", 
            font=("Arial", 12),
            tags="instructions"
        )
    
    def redraw_dots(self):
        # Clear existing dots
        for dot_id in self.dot_ids:
            self.canvas.delete(dot_id)
        self.dot_ids = []
        
        # Redraw dots at current scale
        for x, y in self.dots:
            scaled_x = x * self.scale_factor
            scaled_y = y * self.scale_factor
            dot_size = 10 * max(1, self.scale_factor)  # Scale dot size too
            dot_id = self.canvas.create_oval(
                scaled_x-dot_size, scaled_y-dot_size,
                scaled_x+dot_size, scaled_y+dot_size,
                fill="red", outline="white", width=2
            )
            self.dot_ids.append(dot_id)
    
    def place_dot(self, event, coords=None):
        if coords:
            img_x, img_y = coords
        else:
            # Convert canvas coordinates to original image coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            img_x = canvas_x / self.scale_factor
            img_y = canvas_y / self.scale_factor
        
        # Keep only two dots
        if len(self.dots) >= 2:
            self.dots.pop(0)
        
        # Store original image coordinates
        self.dots.append((img_x, img_y))
        self.redraw_dots()
    
    def clear_dots(self, event=None):
        """Clear all dots from the image (bound to Backspace key)"""
        for dot_id in self.dot_ids:
            self.canvas.delete(dot_id)
        self.dot_ids = []
        self.dots = []

    def finish(self, event):
        self.coords = [(int(x), int(y)) for x, y in self.dots]
        self.root.destroy()

def get_eye_coords(image_path, initial_positions=None):
    """
    Open a GUI to place two red dots on an image, returns their coordinates.
    
    Args:
        image_path (str): Path to PNG image
        
    Returns:
        list: Two (x,y) coordinate tuples [(x1,y1), (x2,y2)] in original image coordinates
    """
    placer = DotPlacer(image_path, initial_positions)
    coords = placer.coords
    
    # Sort by X coordinate before returning
    return sorted(coords, key=lambda c: c[0])
