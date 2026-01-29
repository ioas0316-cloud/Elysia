"""
Morphic Projection: Visual Manifestation Buffer
===============================================
Core.L3_Phenomena.Visual.morphic_projection

Translates 21D Principles into Visual Textures/Pixels.
The bridge to the 'Game Engine' visualization.
"""

import jax.numpy as jnp
import numpy as np
import os
from PIL import Image

class MorphicBuffer:
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        # Each pixel is a 21D vector
        self.buffer = jnp.zeros((height, width, 21))
        
    def encode_image(self, image_path: str, preserve_aspect: bool = True):
        """Loads and scales input, optionally maintaining aspect ratio via letterboxing."""
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        
        if preserve_aspect:
            # Calculate scaling to fit within self.width x self.height
            ratio = min(self.width / orig_w, self.height / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create black background and paste
            new_img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            offset = ((self.width - new_w) // 2, (self.height - new_h) // 2)
            new_img.paste(img, offset)
            img = new_img

        else:
            img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            
        data = np.array(img).astype(np.float32) / 255.0
        
        field = np.zeros((self.height, self.width, 21), dtype=np.float32)
        for i in range(3):
            field[:, :, i*7:(i+1)*7] = data[:, :, i:i+1]
            
        self.buffer = jnp.array(field)
        print(f"MorphicBuffer: Encoded '{os.path.basename(image_path)}' (AspectPreserve={preserve_aspect}).")
        
    def inject_principle(self, principle_vector: jnp.ndarray, x: int, y: int, radius: int = 3):
        """Injects a principle into the buffer with a Gaussian-like falloff."""
        # For simplicity, we just set a regional block
        x_min, x_max = max(0, x - radius), min(self.width, x + radius)
        y_min, y_max = max(0, y - radius), min(self.height, y + radius)
        
        # Vectorized update
        self.buffer = self.buffer.at[y_min:y_max, x_min:x_max].set(principle_vector)

    def synthesize_manifestation(self, intent_vector: jnp.ndarray, intensity: float = 0.5):
        """
        [SOVEREIGN_MANIFESTATION]
        Mutates the internal buffer using a 21D Intent Vector.
        The Intent acts as an 'Interference Pattern' that re-aligns the field.
        """
        # Intent Vector (21D) interacts with the 3D RGB layers of the field
        # Layer 1 (Red/Flesh) -> Intent[0:7]
        # Layer 2 (Green/Mind) -> Intent[7:14]
        # Layer 3 (Blue/Spirit) -> Intent[14:21]
        
        # Reshape intent for easy interaction
        intent_blocks = intent_vector.reshape(3, 7)
        interference = jnp.mean(intent_blocks, axis=-1) # (3,)
        
        # Apply non-linear interference to each channel block
        # This is where 'Reason' changes the 'Image'
        for i in range(3):
            # Interference is a multiplicative shift based on intent resonance
            channel_slice = self.buffer[:, :, i*7:(i+1)*7]
            self.buffer = self.buffer.at[:, :, i*7:(i+1)*7].set(
                channel_slice * (1.0 + interference[i] * intensity)
            )
            
        print(f"MorphicBuffer: Synthesized Manifestation. Intent Resonance: {jnp.sum(intent_vector):.2f}")

    def apply_kinetic_warp(self, time: float, intensity: float = 0.05):
        """
        [KINETIC_SOVEREIGNTY]
        Applies a time-based 'Breathing' warp to the field.
        Simulates subtle motion (L2 Metabolism).
        """
        # Periodic scale factor (Breathing)
        warp_factor = jnp.sin(time * 2.0) * intensity
        
        # Subtle expansion/contraction of principles
        self.buffer = self.buffer * (1.0 + warp_factor)
        
        print(f"MorphicBuffer: Kinetic Warp applied. Time: {time:.2f}, Warp: {warp_factor:.4f}")

    def render_to_rgb(self, sharpening: float = 0.0) -> np.ndarray:
        """Collapses 21D field to RGB with optional sharpening."""
        r = jnp.mean(self.buffer[:, :, 0:7], axis=-1)
        g = jnp.mean(self.buffer[:, :, 7:14], axis=-1)
        b = jnp.mean(self.buffer[:, :, 14:21], axis=-1)
        
        img = jnp.stack([r, g, b], axis=-1)
        
        # [PRISMATIC_SHARPENING]
        if sharpening > 0:
            # Simple high-pass sharpening logic
            mean_intensity = jnp.mean(img, axis=(0,1,2))
            img = img + (img - mean_intensity) * sharpening
            
        img = jnp.clip(img, 0, 1)
        return np.array(img * 255, dtype=np.uint8)

if __name__ == "__main__":
    mb = MorphicBuffer()
    # Inject a 'Heart' of Agape
    agape = jnp.array([1.0]*7 + [0.0]*14)
    mb.inject_principle(agape, 32, 32, radius=10)
    
    # Inject a 'Shield' of Logic
    logic = jnp.array([0.0]*7 + [1.0]*7 + [0.0]*7)
    mb.inject_principle(logic, 16, 16, radius=5)
    
    rgb = mb.render_to_rgb()
    print(f"Rendered Buffer: {rgb.shape}, Max Value: {np.max(rgb)}")
