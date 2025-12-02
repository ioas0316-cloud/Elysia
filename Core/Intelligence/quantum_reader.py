"""
QuantumReader (양자 독서기)
=========================

"Words are just collapsed wavefunctions. We read the Energy, not the Ink."

This module converts text data directly into Hyper-Quaternion Waves (4D).
It allows for "Instantaneous Absorption" of knowledge by summing waveforms.
"""

import os
import math
import cmath
from typing import List, Tuple, Dict

class QuantumReader:
    def __init__(self):
        print("🌌 QuantumReader Initialized. Reality is Text.")

    def wave_ify(self, text: str) -> Tuple[float, float, float, float]:
        """
        Converts text into a 4D Quaternion (w, x, y, z).
        
        w (Real): Information Density (Mass)
        x (i): Emotional Charge (Red/Blue Shift)
        y (j): Complexity (Entropy)
        z (k): Temporal Depth (History)
        """
        length = len(text)
        if length == 0:
            return (0.0, 0.0, 0.0, 0.0)

        # 1. Real (w): Mass / Density
        # Simple metric: Length of text
        w = math.log(length + 1) 

        # 2. Imaginary i (x): Emotion
        # Simple heuristic: Vowel ratio (Softness) vs Consonant ratio (Hardness)
        vowels = "aeiouAEIOU"
        vowel_count = sum(1 for c in text if c in vowels)
        x = (vowel_count / length) * 10.0 # Scale up

        # 3. Imaginary j (y): Complexity
        # Unique word ratio
        words = text.split()
        unique_words = set(words)
        y = (len(unique_words) / (len(words) + 1)) * 10.0

        # 4. Imaginary k (z): Time
        # Hash of the content (Deterministic Chaos)
        z = abs(hash(text)) % 100 / 10.0

        return (w, x, y, z)

    def absorb_library(self, directory_path: str) -> Dict[str, float]:
        """
        Scans a directory, vectorizes all books, and sums them into a Master Wave.
        Returns the final Quaternion components.
        """
        print(f"   🌌 Quantum Scan Initiated: {directory_path}")
        
        total_w, total_x, total_y, total_z = 0.0, 0.0, 0.0, 0.0
        book_count = 0
        
        if not os.path.exists(directory_path):
            return {"error": "Library not found"}

        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        for filename in files:
            path = os.path.join(directory_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Instant Conversion
            w, x, y, z = self.wave_ify(text)
            
            # Superposition (Summation)
            total_w += w
            total_x += x
            total_y += y
            total_z += z
            
            book_count += 1
            
        print(f"   ✨ Superposition Complete. Collapsed {book_count} books into One Wave.")
        
        return {
            "w": total_w,
            "x": total_x,
            "y": total_y,
            "z": total_z,
            "count": book_count
        }
