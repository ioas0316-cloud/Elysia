"""
Monad Gravity: The Physics of Semantic Resonance
================================================

"We do not retrieve; we attract."

This module implements [Axiom Zero], replacing logical search with gravitational attraction.
It calculates the force between Monads based on their 7D resonance and "Narrative Weight".
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np

# 7D Color Spectrum for Vector Mapping
DIMENSIONS = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "INDIGO", "VIOLET"]

class MonadVector:
    """
    Represents the 7D phase-space position of a Monad.
    Values are 0.0 to 1.0 (Intensity of that ray).
    """
    def __init__(self, values: List[float]):
        if len(values) != 7:
            # Pad or truncate if not 7D, but ideally should be 7.
            # For robustness, we resize.
            values = values[:7] + [0.0]*(7-len(values))
        self.vec = np.array(values, dtype=np.float32)

    @property
    def magnitude(self) -> float:
        return np.linalg.norm(self.vec)

    def normalize(self):
        mag = self.magnitude
        if mag == 0: return
        self.vec /= mag

    def distance_to(self, other: 'MonadVector') -> float:
        return np.linalg.norm(self.vec - other.vec)

    def cosine_similarity(self, other: 'MonadVector') -> float:
        mag_a = self.magnitude
        mag_b = other.magnitude
        if mag_a == 0 or mag_b == 0: return 0.0
        return np.dot(self.vec, other.vec) / (mag_a * mag_b)

    def __repr__(self):
        return f"Vec7D({np.round(self.vec, 2)})"

class GravityParticle:
    def __init__(self, id: str, vector: List[float], mass: float = 1.0, content: str = ""):
        self.id = id
        self.pos = MonadVector(vector)
        self.vel = np.zeros(7, dtype=np.float32)
        self.mass = mass
        self.content = content
        self.age = 0
        self.is_anchored = False # If True, does not move (e.g., Core Truths)
        self.causal_links: Dict[str, float] = {} # {target_id: weight} - Causal connection strength

    def add_causal_link(self, target_id: str, weight: float):
        """Register a causal connection (A causes B)."""
        self.causal_links[target_id] = weight

    def apply_force(self, force_vec: np.ndarray):
        if self.is_anchored: return
        # F = ma -> a = F/m
        acc = force_vec / self.mass
        self.vel += acc

    def update(self):
        if self.is_anchored: return
        self.pos.vec += self.vel
        # Friction/Drag (Ether Resistance)
        self.vel *= 0.90
        self.age += 1

class MonadGravityEngine:
    def __init__(self):
        self.particles: Dict[str, GravityParticle] = {}
        self.G = 0.5  # Universal Gravitational Constant
        self.resonance_threshold = 0.85 # Threshold for Genesis
        self.events: List[str] = []
        self.genesis_callback: Optional[Callable[[str, str], None]] = None

    def add_monad(self, id: str, vector: List[float], mass: float = 1.0, content: str = "", anchored: bool = False):
        p = GravityParticle(id, vector, mass, content)
        p.is_anchored = anchored
        self.particles[id] = p
        self.events.append(f"  Monad '{id}' entered the field.")

    def set_genesis_callback(self, callback: Callable[[str, str], None]):
        """Callback function(id1, id2) to trigger when two Monads collide."""
        self.genesis_callback = callback

    def step(self):
        """Run one tick of the universe."""
        ids = list(self.particles.keys())
        processed_pairs = set()

        for i, id1 in enumerate(ids):
            p1 = self.particles[id1]

            for j, id2 in enumerate(ids):
                if i == j: continue
                pair_key = tuple(sorted((id1, id2)))
                if pair_key in processed_pairs: continue
                processed_pairs.add(pair_key)

                p2 = self.particles[id2]

                # 1. Calculate Vector Resonance (Similarity)
                vector_sim = p1.pos.cosine_similarity(p2.pos)

                # 2. Calculate Causal Potential (New)
                # Check if p1 causes p2 or p2 causes p1
                causal_weight = 0.0
                if id2 in p1.causal_links:
                    causal_weight += p1.causal_links[id2]
                if id1 in p2.causal_links:
                    causal_weight += p2.causal_links[id1]

                # Total Resonance = (Vector * 0.3) + (Causal * 0.7)
                # Meaning is defined by consequence, not just appearance.
                total_resonance = (vector_sim * 0.3) + (causal_weight * 0.7)

                # 3. Apply Gravity
                # Force = G * (Resonance^3) / Distance^2
                dist = p1.pos.distance_to(p2.pos)
                dist = max(dist, 0.1)

                force_magnitude = (self.G * (total_resonance**3) * p1.mass * p2.mass) / (dist**2)

                # Vector from p1 to p2
                direction = p2.pos.vec - p1.pos.vec
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm

                # Apply Forces
                p1.apply_force(direction * force_magnitude)
                p2.apply_force(-direction * force_magnitude)

                # 4. Check for Genesis (Collision/Fusion)
                # If very close AND highly resonant
                if dist < 0.5 and total_resonance > self.resonance_threshold:
                    self._trigger_genesis(p1, p2)

        # Update positions
        for p in self.particles.values():
            p.update()

    def _trigger_genesis(self, p1: GravityParticle, p2: GravityParticle):
        self.events.append(f"  GENESIS: '{p1.id}' and '{p2.id}' are fusing!")
        if self.genesis_callback:
            self.genesis_callback(p1.id, p2.id)

            # Repel them slightly after fusion to prevent infinite loop
            # "The Big Bang push"
            direction = p2.pos.vec - p1.pos.vec
            p1.vel -= direction * 0.5
            p2.vel += direction * 0.5

    def get_top_events(self, n=5) -> List[str]:
        res = self.events[-n:]
        # self.events.clear() # Keep history? Or clear? Let's keep for now but maybe clear in production.'
        return res

    def rotate_manifold(self, angle: float, axis_pair: Tuple[int, int] = (0, 1)):
        """
        [Perspective Manifold]
        Rotates the entire 7D semantic space along a specific plane (2 axes).
        This simulates 'changing the point of view'.
        """
        i, j = axis_pair
        c, s = np.cos(angle), np.sin(angle)

        # Apply rotation to all particles
        for p in self.particles.values():
            # Standard 2D rotation matrix application on selected axes
            xi, xj = p.pos.vec[i], p.pos.vec[j]
            p.pos.vec[i] = c * xi - s * xj
            p.pos.vec[j] = s * xi + c * xj

    def find_optimal_perspective(self, target_id: str, candidates: List[str]) -> Dict[str, Any]:
        """
        [Axis-Shifting]
        Iteratively rotates the manifold to find the angle where the target
        aligns best (highest resonance) with the candidates.
        """
        if target_id not in self.particles:
            return {"status": "Target Not Found", "angle": 0.0, "improvement": 0.0}

        target_p = self.particles[target_id]
        candidate_ps = [self.particles[cid] for cid in candidates if cid in self.particles]

        if not candidate_ps:
            return {"status": "No Candidates", "angle": 0.0}

        best_angle = 0.0
        max_resonance = 0.0

        # Initial Resonance (Angle 0)
        current_resonance = sum(target_p.pos.cosine_similarity(cp.pos) for cp in candidate_ps)
        max_resonance = current_resonance

        # Try rotating 0 to 90 degrees in 10 steps on Red-Orange plane (0, 1)
        # In full implementation, we would optimize over all 7D planes (21 combinations).

        # [Correction] We rotate ONLY the Target relative to the Candidates.
        # Rotating the whole universe preserves relative distance (Isometry).
        # To find a new perspective, we must tilt the Object (Target) to fit the Slot (Context).

        original_vec = target_p.pos.vec.copy()

        for angle in np.linspace(0, np.pi/2, 10):
            c, s = np.cos(angle), np.sin(angle)
            i, j = (0, 1) # Red-Orange Plane

            # Rotate temporary target vector
            xi, xj = original_vec[i], original_vec[j]

            # Apply rotation to target only
            rotated_vec = original_vec.copy()
            rotated_vec[i] = c * xi - s * xj
            rotated_vec[j] = s * xi + c * xj

            # Create temp monad for calculation
            temp_pos = MonadVector(list(rotated_vec))

            sim_resonance = sum(temp_pos.cosine_similarity(cp.pos) for cp in candidate_ps)

            if sim_resonance > max_resonance:
                max_resonance = sim_resonance
                best_angle = angle

        return {
            "status": "Perspective Optimized",
            "best_angle_deg": math.degrees(best_angle),
            "original_resonance": current_resonance,
            "optimized_resonance": max_resonance,
            "improvement_factor": max_resonance / (current_resonance + 1e-9)
        }
