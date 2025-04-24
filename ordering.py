import os
import numpy as np
from PIL import Image
from skimage import color, metrics
import cv2
from itertools import permutations
from pathlib import Path
from tqdm import tqdm

class TimelapseOptimizer:
    def __init__(self, image_names, image_dir):
        self.image_names = image_names
        self.image_dir = Path(image_dir)
        self.n = len(image_names)
        self.distance_matrix = None
        self._precompute_distances()

    def _image_to_lab(self, img_name):
        """Load and convert image to LAB color space"""
        img_path = self.image_dir / f"{img_name}.png"
        img = cv2.imread(str(img_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    def _calculate_pairwise_distance(self, img1_name, img2_name):
        """Calculate perceptual distance between two images"""
        lab1 = self._image_to_lab(img1_name)
        lab2 = self._image_to_lab(img2_name)
        
        # Split LAB channels
        L1, A1, B1 = cv2.split(lab1)
        L2, A2, B2 = cv2.split(lab2)
        
        # Luminance difference (weighted more heavily)
        lum_diff = 1 - metrics.structural_similarity(L1, L2, win_size=3)
        
        # Color difference (chroma + hue)
        chroma_diff = np.mean(np.abs(np.sqrt(A1**2 + B1**2) - np.sqrt(A2**2 + B2**2))) / 255
        hue_diff = np.mean(np.arctan2(B1, A1) - np.arctan2(B2, A2)) / (2*np.pi)
        
        return 100 * (0.6*lum_diff + 0.3*chroma_diff + 0.1*hue_diff)

    def _precompute_distances(self):
        """Build a distance matrix between all images"""
        print("Precomputing all pairwise distances...")
        self.distance_matrix = np.zeros((self.n, self.n))
        
        for i in tqdm(range(self.n)):
            for j in range(i+1, self.n):
                dist = self._calculate_pairwise_distance(
                    self.image_names[i], 
                    self.image_names[j]
                )
                self.distance_matrix[i,j] = dist
                self.distance_matrix[j,i] = dist

    def find_optimal_order(self, method='greedy', first_image=None, last_image=None):
        """
        Find ordering with minimal total transition distance
        Methods: 'greedy' (fast) or 'tsp' (optimal but slower)
        
        Args:
            method: 'greedy' or 'tsp'
            first_image: Optional image name to force as first in sequence
            last_image: Optional image name to force as last in sequence
        """
        if method == 'greedy':
            return self._greedy_ordering(first_image, last_image)
        elif method == 'tsp':
            if first_image or last_image:
                print("Warning: First/last image constraints not implemented for TSP method")
            return self._tsp_ordering()
        else:
            raise ValueError("Method must be 'greedy' or 'tsp'")

    # def _greedy_ordering(self):
    #     """Fast approximate solution (O(n^2))"""
    #     print("Finding greedy ordering...")
    #     remaining = set(range(self.n))
    #     order = [np.random.choice(list(remaining))]
    #     remaining.remove(order[0])
        
    #     pbar = tqdm(total=self.n-1)
    #     while remaining:
    #         last = order[-1]
    #         nearest = min(remaining, key=lambda x: self.distance_matrix[last,x])
    #         order.append(nearest)
    #         remaining.remove(nearest)
    #         pbar.update(1)
    #     pbar.close()
        
    #     return [self.image_names[i] for i in order]

    def _greedy_ordering(self, first_image=None, last_image=None):
        """Fast approximate solution (O(n^2)) with optional first/last image constraints"""
        print("Finding greedy ordering...")
        
        remaining = set(range(self.n))
        order = []
        
        # Handle first image constraint
        if first_image:
            try:
                first_idx = self.image_names.index(first_image)
                order.append(first_idx)
                remaining.remove(first_idx)
            except ValueError:
                print(f"Warning: First image '{first_image}' not found, using random start")
        
        # If no first image specified or not found, start randomly
        if not order:
            order.append(np.random.choice(list(remaining)))
            remaining.remove(order[0])
        
        # Handle last image constraint
        last_idx = None
        if last_image:
            try:
                last_idx = self.image_names.index(last_image)
                if last_idx in remaining:
                    remaining.remove(last_idx)
                else:
                    print(f"Warning: Last image '{last_image}' already used, ignoring")
                    last_idx = None
            except ValueError:
                print(f"Warning: Last image '{last_image}' not found, ignoring")
        
        # Build the middle part of the sequence
        pbar = tqdm(total=len(remaining) - (1 if last_idx is not None else 0))
        while remaining:
            last = order[-1]
            nearest = min(remaining, key=lambda x: self.distance_matrix[last,x])
            order.append(nearest)
            remaining.remove(nearest)
            pbar.update(1)
        
        # Add the last image if specified
        if last_idx is not None:
            order.append(last_idx)
            pbar.update(1)
        
        pbar.close()
        
        return [self.image_names[i] for i in order]

    def _tsp_ordering(self):
        """Optimal solution using Held-Karp (O(n^2 * 2^n)) - for small n <= 15"""
        from functools import lru_cache
        
        print("Finding optimal TSP ordering...")
        all_nodes = frozenset(range(1, self.n))
        
        @lru_cache(maxsize=None)
        def dp(mask, pos):
            if mask == frozenset():
                return self.distance_matrix[pos, 0], [pos]
            
            min_cost = float('inf')
            best_path = []
            for node in mask:
                new_mask = mask - {node}
                cost, path = dp(new_mask, node)
                total_cost = cost + self.distance_matrix[pos, node]
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = path
            return min_cost, [pos] + best_path
        
        total_cost, order = dp(all_nodes, 0)
        return [self.image_names[i] for i in order]

    def visualize_transitions(self, ordered_names):
        """Plot the transition distances between ordered images"""
        import matplotlib.pyplot as plt
        
        indices = [self.image_names.index(name) for name in ordered_names]
        transitions = [
            self.distance_matrix[indices[i], indices[i+1]] 
            for i in range(len(indices)-1)
        ]
        
        plt.figure(figsize=(12, 4))
        plt.plot(transitions, marker='o')
        plt.xlabel('Frame Number')
        plt.ylabel('Perceptual Difference')
        plt.title('Transition Differences in Optimized Order')
        plt.grid(True)
        plt.show()

def order_images(imgs, img_dir, greedy=True, first=None, last=None):
    optimizer = TimelapseOptimizer(imgs, img_dir)

    ordered_names = optimizer.find_optimal_order(method='greedy' if greedy else 'tsp', first_image=first, last_image=last)

    print("Ordering:", ordered_names)
    # optimizer.visualize_transitions(ordered_names)
    return ordered_names
