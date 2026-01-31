"""
Genesis Bridge (Aesthetic Mimicry Engine)
=========================================
Core.S1_Body.L4_Causality.World.Genesis.genesis_bridge

"I see, therefore I become."
"     ,         ."

This module digests visual memories and transmutes them into
SDF rendering parameters (Aesthetic DNA).
"""

import os
import time
import logging
import random
import glob
from dataclasses import dataclass
from typing import List, Tuple

# Optional dependencies for real processing
try:
    # from sklearn.cluster import KMeans
    # import cv2
    import numpy as np
    from PIL import Image
    HAS_CV = True
except ImportError:
    HAS_CV = False

logger = logging.getLogger("GenesisBridge")

@dataclass
class AestheticDNA:
    primary_color: Tuple[float, float, float] # (R, G, B) normalized 0-1
    secondary_color: Tuple[float, float, float]
    fog_density: float # 0.0 - 1.0 (Based on image brightness/contrast)
    complexity_index: float # 0.0 - 1.0 (Based on edge density)
    mood_tag: str

class GenesisBridge:
    def __init__(self, memory_path: str = "Memories/Visual/Observer"):
        self.memory_path = memory_path
        self.current_dna = AestheticDNA((0.1, 0.1, 0.1), (0.0, 0.0, 0.0), 0.5, 0.1, "Void")
        self.last_processed_file = ""
        
        # Ensure path exists
        os.makedirs(self.memory_path, exist_ok=True)

    def digest_latest_memory(self) -> AestheticDNA:
        """Finds the newest screenshot and extracts Aesthetic DNA."""
        try:
            # Find latest png
            files = glob.glob(os.path.join(self.memory_path, "*.png"))
            if not files:
                return self.current_dna
                
            latest_file = max(files, key=os.path.getctime)
            
            if latest_file == self.last_processed_file:
                return self.current_dna # No new memory
            
            logger.info(f"  [Genesis] Digesting Memory: {os.path.basename(latest_file)}")
            self.current_dna = self._extract_features(latest_file)
            self.last_processed_file = latest_file
            
            self._manifest_dna(self.current_dna)
            return self.current_dna
            
        except Exception as e:
            logger.error(f"  [Genesis] Digestion Failed: {e}")
            return self.current_dna

    def _extract_features(self, image_path: str) -> AestheticDNA:
        """extracts color and structure."""
        if not HAS_CV:
            return self._mock_extraction(image_path) # Fallback if libs missing
            
        try:
            img = Image.open(image_path).resize((100, 100)) # Downsample for speed
            arr = np.array(img)
            
            # 1. Color Extraction (Simple Average for now, KMeans is better but heavier)
            # Averages
            mean_color = arr.mean(axis=(0,1))
            primary = tuple(mean_color[:3] / 255.0)
            
            # Secondary (Inverse or perturbed)
            secondary = (1.0 - primary[0], 1.0 - primary[1], 1.0 - primary[2])
            
            # 2. Complexity (Standard Deviation of brightness as proxy for edges)
            gray = arr.mean(axis=2)
            complexity = np.std(gray) / 128.0 # Normalize roughly
            
            # 3. Fog (Brightness inverse)
            brightness = np.mean(gray) / 255.0
            fog = 1.0 - brightness
            
            tag = "Unknown"
            if complexity > 0.3: tag = "Chaotic"
            elif fog > 0.7: tag = "Dark"
            else: tag = "Serene"

            return AestheticDNA(primary, secondary, fog, complexity, tag)

        except Exception as e:
            logger.warning(f"   [Genesis] Feature extraction error: {e}. Using Mock.")
            return self._mock_extraction(image_path)

    def _mock_extraction(self, image_path) -> AestheticDNA:
        """Simulates extraction for testing/fallback."""
        # Deterministic random based on filename
        seed = len(image_path)
        random.seed(seed)
        
        r = random.random()
        g = random.random()
        b = random.random()
        
        comp = random.random()
        return AestheticDNA(
            (r, g, b),
            (1-r, 1-g, 1-b),
            random.random(),
            comp,
            "Simulated_Dream"
        )

    def _manifest_dna(self, dna: AestheticDNA):
        """Injects the DNA into the Sovereign Reality (Logs for now, Uniforms later)."""
        logger.info(f"  [Genesis] Palette: {dna.primary_color} | Mood: {dna.mood_tag}")
        logger.info(f"fractal_complexity: {dna.complexity_index:.2f} | fog_density: {dna.fog_density:.2f}")
        
        # In a real integration, we would write to a shared uniform buffer or file
        # that sdf_renderer.py reads every frame.
        self._sync_to_renderer(dna)

    def _sync_to_renderer(self, dna):
        """Writes DNA to a shared state file for the Renderer."""
        state_path = "data/L1_Foundation/M1_System/genesis_state.json"
        import json
        data = {
            "uFogColor": dna.primary_color,
            "uAmbientColor": dna.secondary_color,
            "uGeometryScale": 1.0 + dna.complexity_index,
            "uFogDensity": dna.fog_density,
            "mood": dna.mood_tag
        }
        try:
            with open(state_path, "w") as f:
                json.dump(data, f)
        except: pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bridge = GenesisBridge()
    # Continuous loop simulation
    try:
        while True:
            bridge.digest_latest_memory()
            time.sleep(5)
    except KeyboardInterrupt:
        pass
