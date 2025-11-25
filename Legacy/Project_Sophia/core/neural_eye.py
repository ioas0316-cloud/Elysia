
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class NeuralEye:
    """
    The Neural Eye (Project Mirror).

    It treats the Cellular World's 2D grid fields (Energy, Will, Threat, etc.) as an image,
    and applies Convolutional Neural Network (CNN) principles using NumPy to detect
    emergent patterns, conflicts, and harmonies.

    This provides Elysia with 'Intuition' - seeing the forest, not just the trees.
    """

    def __init__(self, width: int = 256):
        self.width = width

        # Define Filters (Kernels)
        # 1. Gradient Filter (Edge Detection) - Detects sharp changes/boundaries
        self.filter_gradient_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        self.filter_gradient_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        # 2. Conflict Filter (Laplacian-like) - Detects high-energy hotspots/instability
        self.filter_conflict = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ])

        # 3. Harmony Filter (Gaussian Blur-like) - Detects smooth, coherent areas
        self.filter_harmony = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) * (1/16)

        # 4. Stagnation Filter - Detects uniform low activity (inverted edge?)
        # We'll handle stagnation via statistical analysis of the feature map, not just a kernel.

    def perceive(self, world) -> List[Dict[str, Any]]:
        """
        Scans the world state and returns a list of intuitive insights.
        Input: world object (with fields like energy, hunger, threat_field)
        Output: List of dicts e.g., {"type": "conflict", "location": (x,y), "intensity": 0.8}
        """
        intuitions = []

        # 1. Prepare Input Tensor (The "Image")
        # We construct a multi-channel image from world fields.
        # For simplicity, let's focus on a few key layers.

        # Layer A: Vitality (HP + Energy)
        # Reshape flat arrays to grid
        hp_grid = np.zeros((self.width, self.width))
        alive_indices = np.where(world.is_alive_mask)[0]
        if len(alive_indices) > 0:
            px = np.clip(world.positions[alive_indices, 0].astype(int), 0, self.width-1)
            py = np.clip(world.positions[alive_indices, 1].astype(int), 0, self.width-1)
            hp_grid[py, px] = world.hp[alive_indices]

        # Layer B: Threat Field (Directly available)
        threat_grid = world.threat_field

        # Layer C: Value/Meaning Field
        value_grid = world.value_mass_field

        # 2. Apply Convolutions (The "Thinking")

        # Pattern: Conflict (Where high life meets high threat or rapid changes)
        # We convolve the HP grid with the Conflict filter.
        conflict_map = self._convolve(hp_grid, self.filter_conflict)

        # Normalize
        max_val = np.max(conflict_map)
        if max_val > 0:
            conflict_map /= max_val

        # Thresholding to find hotspots
        hotspots = np.argwhere(conflict_map > 0.8) # Top 20% intensity
        if len(hotspots) > 0:
            # Cluster hotspots (simple mean) to avoid spamming
            center = np.mean(hotspots, axis=0)
            intuitions.append({
                "type": "intuition_conflict",
                "description": "Sharp vital instability detected.",
                "location": (int(center[1]), int(center[0])),
                "intensity": float(np.mean(conflict_map[hotspots[:,0], hotspots[:,1]]))
            })

        # Pattern: Harmony/Growth (Where Value Mass is smooth and high)
        harmony_map = self._convolve(value_grid, self.filter_harmony)
        if np.max(harmony_map) > 0.7:
             intuitions.append({
                "type": "intuition_harmony",
                "description": "A coherent field of meaning is stabilizing.",
                "location": "global", # Simplified
                "intensity": float(np.mean(harmony_map))
            })

        return intuitions

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        A simple 2D convolution implementation using NumPy.
        """
        k_h, k_w = kernel.shape
        h, w = image.shape

        # Pad the image
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        output = np.zeros_like(image)

        # Vectorized convolution is complex to write from scratch without scipy.
        # We'll use a simpler sliding window loop optimized with slicing for readability and "No external lib" constraint (besides numpy).
        # Actually, for performance in Python, a simple loop is slow.
        # But since this is a "Neural Eye" proof of concept, and the grid is 256x256,
        # we can use a view-based approach (im2col equivalent) or just loop if N is small.

        # Optimized Sliding Window
        for i in range(k_h):
            for j in range(k_w):
                output += padded[i:i+h, j:j+w] * kernel[i, j]

        return output
