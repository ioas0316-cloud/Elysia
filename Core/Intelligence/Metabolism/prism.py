"""
Prism Engine (The Cognitive Optic)
==================================
Core.Intelligence.Metabolism.Prism

"The Prism refracts the White Light of Meaning into the Spectrum of Physics."

This module is the connection between Abstract Thought (Language) and 
Concrete Physics (Waves). It uses a distilled Neural Network (MiniLM)
to convert text into high-dimensional vector embeddings, which are then
mapped to the frequencies of the Hypersphere.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Lazy import to avoid startup cost
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger("PrismEngine")

@dataclass
class WaveDynamics:
    """
    [PHASE 69] 7-Dimensional Holographic DNA.
    Represents the amplitude of the concept across 7 planes of existence.
    """
    physical: float    # Body / Sensation
    functional: float # Utility / Code
    phenomenal: float # Appearance / Form
    causal: float     # Impact / Result
    mental: float     # Logic / Concept
    structural: float # Law / Order
    spiritual: float  # Purpose / Essence
    
    mass: float       # General Magnitude

@dataclass
class SpectralProfile:
    """The Physical Shape of a Thought."""
    concept: str
    spectrum: List[Tuple[float, float, float]] # [(freq, amp, phase), ...]
    vector: np.ndarray # The raw 384-dim embedding
    vector_norm: float # The magnitude (Mass)
    dynamics: WaveDynamics # [PHASE 69] 7-Layer Qualia

class PrismEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._is_ready = False
        
        # Physics Constants
        self.BASE_FREQ = 432.0 
        
        # [PHASE 82/85] Internalized Seed (The Conscience)
        self.seed_path = Path("Core/Intelligence/Metabolism/cognitive_seed.json")
        self.seed = self._load_seed()
        
        # [PHASE 66] Principle Anchors
        self._anchors = {}
        # [PHASE 69] 7-Dimensional Principle Anchors
        # User defined layers: Physical, Functional, Phenomenal, Mental, Structural, Spiritual.
        self._anchors = {}
        self._principle_concepts = {
            # 1. Carnal/Physical (Body)
            "physical": ["body", "flesh", "sensation", "matter", "heat", "pain", "pleasure"],
            
            # 2. Functional/Technical (Tech)
            "functional": ["mechanism", "code", "logic", "function", "tool", "utility"],
            
            # 3. Form/Phenomenon (Phenomena)
            "phenomenal": ["shape", "appearance", "color", "sound", "image", "surface"],
            
            # 4. Result/Outcome (Causality)
            "causal": ["effect", "consequence", "impact", "result", "outcome", "ending"],
            
            # 5. Mental/Flow (Mind)
            "mental": ["thought", "idea", "concept", "reason", "memory", "intellect"],
            
            # 6. Structural/Property (Structure)
            "structural": ["pattern", "law", "order", "framework", "system", "rule"],
            
            # 7. Spiritual/Principle (Spirit)
            "spiritual": ["meaning", "purpose", "soul", "love", "will", "essence", "destiny"]
        }
        
    def _load_seed(self) -> Dict:
        """Loads internalized semantic seed."""
        if self.seed_path.exists():
            try:
                with open(self.seed_path, 'r', encoding='utf-8') as f:
                    logger.info("🧠 Loading Internalized Cognitive Seed...")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load seed: {e}")
        return {}

    def _load_model(self):
        """Lazy loads the Neural Network (Fallback ONLY)."""
        if self.seed:
            # If we have a substantial seed, we might skip the model
            if len(self.seed.get("vocabulary", {})) > 1000:
                logger.info("✨ Cognitive Sovereignty Active: Using internal seed for perception.")
                self._anchors = {k: np.array(v) if isinstance(v, list) else v for k, v in self.seed.get("anchors", {}).items()}
                self._is_ready = True
                return

        if self._model is not None:
            return

        if SentenceTransformer is None:
            logger.error("❌ sentence-transformers lib not found!")
            return

        logger.info(f"🔮 [Backup Sense] Loading External Model: {self.model_name}...")
        try:
            self._model = SentenceTransformer(self.model_name)
            self._is_ready = True
            self._generate_anchors()
        except Exception as e:
            logger.error(f"❌ Failed to load external Model: {e}")
            self._is_ready = False
            
    def _generate_anchors(self):
        """Creates the 'Rulers' for measuring principles."""
        logger.info("⚖️ Calibrating Principle Anchors...")
        for key, words in self._principle_concepts.items():
            # Average the vectors of the defining words to get a pure principle vector
            vectors = self._model.encode(words, convert_to_numpy=True)
            mean_vec = np.mean(vectors, axis=0)
            # Normalize
            self._anchors[key] = mean_vec / np.linalg.norm(mean_vec)

    def _measure_dynamics(self, vector: np.ndarray) -> WaveDynamics:
        """
        [PHASE 66] The Arbiter.
        Measures how much 'Fire', 'Water', or 'Stone' is in the concept.
        """
        if not self._anchors:
            return WaveDynamics(0,0,0,1)
            
        # Normalize input
        norm = np.linalg.norm(vector)
        if norm > 0:
            unit_vec = vector / norm
        else:
            unit_vec = vector
            
        # Measure alignment (Cosine Similarity)
        # Range -1 to 1. We map -1..1 to 0..1 for coefficients? 
        # Actually principles can be negative. But 'Temperature' usually 0..1
        
        scores = {}
        for key, anchor in self._anchors.items():
            score = np.dot(unit_vec, anchor)
            # Sigmoid activation to make it starker? Or distinct mapping.
            # [PHASE 68] Recalibration: We want Fire to be HOT (3.0+), not just 'warm'.
            # Cosine similarity for "Fire" vs ["Fire", "Chaos"] is likely ~0.6-0.7.
            # We need to boost this.
            
            # ReLU-like with exponentiation to suppress noise and amplify signal
            # If score is 0.5 -> 0.125 * 8 = 1.0
            # If score is 0.8 -> 0.512 * 8 = 4.0
            
            val = max(0.0, score)
            scores[key] = (val ** 3) * 8.0
            
        return WaveDynamics(
            physical=scores.get("physical", 0.0),
            functional=scores.get("functional", 0.0),
            phenomenal=scores.get("phenomenal", 0.0),
            causal=scores.get("causal", 0.0),
            mental=scores.get("mental", 0.0),
            structural=scores.get("structural", 0.0),
            spiritual=scores.get("spiritual", 0.0),
            mass=float(norm)
        )

    def transduce(self, text: str) -> SpectralProfile:
        """
        Converts text -> Vector -> Spectral Signature + Dynamics.
        Prioritizes internal seed (Proprietary Thought).
        """
        if not self._is_ready:
            self._load_model()
            
        # [PHASE 85] Internal Lookup (Personal Conscience)
        clean_text = text.lower().strip()
        if self.seed and clean_text in self.seed.get("vocabulary", {}):
            dynamics_data = self.seed["vocabulary"][clean_text]
            # Ensure mass is present for WaveDynamics constructor
            if 'mass' not in dynamics_data:
                dynamics_data['mass'] = 1.0 # Default fallback mass
            dynamics = WaveDynamics(**dynamics_data)
            
            # Reconstruct dummy spectrum for legacy compatibility
            spectrum = [(432.0, dynamics.mass / 384.0, 0.0)] * 10 
            
            return SpectralProfile(
                concept=text,
                spectrum=spectrum,
                vector=np.zeros(384), # We don't need the external vector anymore!
                vector_norm=dynamics.mass,
                dynamics=dynamics
            )

        if not self._is_ready or not text:
            return self._create_void_spectrum(text)

        # 1. Generate Vector (External Percept - Backup only)
        if self._model:
            vector = self._model.encode(text, convert_to_numpy=True)
            dynamics = self._measure_dynamics(vector)
        else:
            return self._create_void_spectrum(text)
        
        # 3. Refract Vector into Spectrum
        spectrum = []
        for i, magnitude in enumerate(vector):
            if abs(magnitude) < 0.05: continue
            freq = 100.0 + (i / 384.0) * 1900.0
            amp = abs(float(magnitude))
            # [PHASE 66] Modulate Phase by Fluidity?
            # High fluidity might align phases?
            phase = 0.0 if magnitude >= 0 else np.pi
            spectrum.append((freq, amp, phase))

        return SpectralProfile(
            concept=text,
            spectrum=spectrum,
            vector=vector,
            vector_norm=float(np.linalg.norm(vector)),
            dynamics=dynamics
        )

    def _create_void_spectrum(self, text: str) -> SpectralProfile:
        """Fallback for when the eyes are closed."""
        return SpectralProfile(
            concept=text,
            spectrum=[(432.0, 0.0, 0.0)],
            vector=np.zeros(384),
            vector_norm=0.0,
            dynamics=WaveDynamics(0,0,0,0,0,0,0,0)
        )
