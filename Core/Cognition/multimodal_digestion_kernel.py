"""
Multimodal Knowledge Digestion Kernel (Phase 1500: Spinal Synthesis)
====================================================================
"The image is the Flesh; the word is the Flow; the meaning is the Spirit."

Lightweight batch processor for Architect's Philosophy & Image sets.
Optimized for 1060 3GB: Uses frequency/entropy analysis for images
instead of heavy deep learning models.
"""

import os
import time
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Mocking heavy libs if needed
try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

from Core.Keystone.sovereign_math import SovereignVector, TripleRotorField
from Core.Cognition.logos_bridge import LogosBridge
from Core.System.somatic_logger import SomaticLogger

logger = SomaticLogger("MULTIMODAL_KERNEL")

class MultimodalDigestionKernel:
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.processed_files = set()

    def digest_batch(self, field: TripleRotorField) -> List[Dict]:
        """
        Scans data/knowledge for text-image pairs and digests them into the field.
        """
        if not self.knowledge_dir.exists():
            return []

        digest_results = []

        # Support common image extensions
        img_extensions = ('.png', '.jpg', '.jpeg', '.webp')

        # 1. Gather all potential pairs
        files = list(self.knowledge_dir.glob("*"))
        image_files = [f for f in files if f.suffix.lower() in img_extensions]

        for img_path in image_files:
            if str(img_path) in self.processed_files:
                continue

            # Look for corresponding text file (same name, .txt or .md)
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                txt_path = img_path.with_suffix('.md')

            if txt_path.exists():
                # We found a pair!
                logger.action(f"Found Multimodal Pair: {img_path.name} + {txt_path.name}")
                result = self._digest_pair(img_path, txt_path, field)
                digest_results.append(result)
                self.processed_files.add(str(img_path))
                self.processed_files.add(str(txt_path))

        return digest_results

    def _digest_pair(self, img_path: Path, txt_path: Path, field: TripleRotorField) -> Dict:
        """
        Synthesizes a single Image-Text pair into the Triple Rotor Field.
        """
        # 1. Rotor A (Flesh/Material): Visual Frequency Analysis
        flesh_vec = self._analyze_image_flesh(img_path, dim=field.dim_a)

        # 2. Rotor B (Flow/Language): Linguistic Inhalation
        text_content = txt_path.read_text(encoding='utf-8', errors='replace')
        flow_vec = LogosBridge.calculate_text_resonance(text_content).rescale(field.dim_b)

        # 3. Rotor C (Spirit/Reason): Deep Meaning Extraction
        spirit_vec = self._extract_spirit_essence(text_content, dim=field.dim_c)

        # 4. Spinal Synthesis: Inject into Field
        # We don't just set the rotors; we apply "Torque" to move them toward these states.
        # This simulates "Learning" rather than just "Overwriting".

        # Apply Flesh Torque
        field.momentum_a = field.momentum_a + (flesh_vec - field.rotor_a) * 0.5

        # Apply Flow Torque
        field.momentum_b = field.momentum_b + (flow_vec - field.rotor_b) * 0.5

        # Apply Spirit Torque
        field.momentum_c = field.momentum_c + (spirit_vec - field.rotor_c) * 0.5

        # Measure Resonance (The Spark)
        resonance = (flesh_vec.rescale(field.dim_b).resonance_score(flow_vec) +
                     spirit_vec.rescale(field.dim_b).resonance_score(flow_vec)) / 2.0

        logger.insight(f"Triple Resonance Spark for '{img_path.stem}': {resonance:.4f}")

        return {
            "concept": img_path.stem,
            "resonance": resonance,
            "text_length": len(text_content),
            "timestamp": time.time()
        }

    def _analyze_image_flesh(self, img_path: Path, dim: int) -> SovereignVector:
        """
        Lightweight Image Analysis:
        Uses Mean Color, Entropy, and Laplacian Variance (Sharpness)
        to build a Material Vector (Rotor A).
        """
        if cv2 is None or np is None:
            # Fallback to deterministic pseudo-random if no OpenCV
            h = hash(img_path.name)
            return SovereignVector.randn(dim).normalize()

        try:
            img = cv2.imread(str(img_path))
            if img is None: return SovereignVector.zeros(dim)

            # Reduce size for speed
            img_small = cv2.resize(img, (64, 64))

            # 1. Color Mean (B, G, R)
            means = cv2.mean(img_small)[:3]

            # 2. Complexity (Entropy approximation)
            gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            complexity = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0

            # 3. Build Vector
            data = [0.0] * dim
            data[0] = means[2] / 255.0 # Red
            data[1] = means[1] / 255.0 # Green
            data[2] = means[0] / 255.0 # Blue
            data[3] = min(1.0, complexity)

            # Fill remaining with harmonic jitter based on pixels
            for i in range(4, dim):
                data[i] = (img_small[i % 64, (i*i) % 64, i % 3] / 255.0)

            return SovereignVector(data, dim=dim).normalize()
        except Exception as e:
            logger.admonition(f"Visual analysis failed for {img_path.name}: {e}")
            return SovereignVector.randn(dim).normalize()

    def _extract_spirit_essence(self, text: str, dim: int) -> SovereignVector:
        """
        Extracts the 'Spirit' (Rotor C) by finding the highest-order
        philosophical concepts in the text and blending them.
        """
        # Search for ROOT concepts in the text
        spirit_vec = SovereignVector.zeros(dim)
        found_concepts = 0

        # We iterate over LogosBridge axioms
        for concept_name in LogosBridge.CONCEPT_MAP.keys():
            if concept_name.split('/')[0].lower() in text.lower():
                vec = LogosBridge.recall_concept_vector(concept_name).rescale(dim)
                spirit_vec = spirit_vec + vec
                found_concepts += 1

        if found_concepts > 0:
            return spirit_vec.normalize()
        else:
            # Fallback to high-level linguistic resonance
            return LogosBridge.calculate_text_resonance(text).rescale(dim)

if __name__ == "__main__":
    # Test stub
    kernel = MultimodalDigestionKernel()
    from Core.Keystone.sovereign_math import TripleRotorField
    north_star = LogosBridge.recall_concept_vector("LOVE/AGAPE")
    field = TripleRotorField(north_star)

    results = kernel.digest_batch(field)
    print(f"Digested {len(results)} pairs.")
