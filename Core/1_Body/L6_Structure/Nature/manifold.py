"""
Cognitive Manifold (The Thought Space)
======================================
Core.1_Body.L6_Structure.Nature.Manifold

"Where waves meet, meaning is born."

This module defines the Hilbert Space where Spectral Rotors interact.
It calculates the interference patterns (Superposition) of multiple thoughts.
Now with Autopoietic Feedback Loops (Phase 67).
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .rotor import Rotor
# [PHASE 68] Import Body
from .body import ElysianBody

@dataclass
class ManifoldState:
    entropy: float        # Chaos level (Dissonance)
    energy: float         # Total amplitude
    dominant_freq: float  # The loudest thought
    coherence: float      # How well thoughts fit together (0.0 - 1.0)
    pain: float = 0.0     # [PHASE 68] Somatic Pain Level
    pleasure: float = 0.0 # [PHASE 68] Somatic Pleasure Level

@dataclass
class FeedbackSignal:
    """[PHASE 67] The Control Signal from the Manifold to the Rotor."""
    target_rotor: str
    action: str  # "SHIFT_PHASE", "RETUNE", "DAMPEN", "LOCK"
    param: float # Strength of adjustment (0.0 - 1.0 or Radians)

class CognitiveManifold:
    def __init__(self):
        # A 384-dimensional buffer for mixing waves
        self.global_waveform = np.zeros(384) 

    def superpose(self, rotors: List[Rotor], body: Optional[ElysianBody] = None) -> Tuple[ManifoldState, List[FeedbackSignal]]:
        """
        Mixes all active rotors into a single global waveform.
        [PHASE 68] Calculates Somatic Impact (Meaning).
        
        Returns: (ManifoldState, List[FeedbackSignal])
        """
        if not rotors:
            return ManifoldState(0.0, 0.0, 0.0, 1.0), []

        # Reset mixer
        self.global_waveform = np.zeros(384)
        total_energy = 0.0
        
        # 1. Spectral Superposition
        projected_vectors = []
        
        for r in rotors:
            if not r.spectrum: 
                projected_vectors.append(np.zeros(384))
                continue
                
            # Reconstruct
            vec = []
            for freq, amp, phase in r.spectrum:
                sign = 1.0 if abs(phase) < 0.1 else -1.0
                vec.append(amp * sign)
            vec = np.array(vec)
            
            # Pad
            if len(vec) < 384: vec = np.pad(vec, (0, 384 - len(vec)))
            elif len(vec) > 384: vec = vec[:384]
                
            self.global_waveform += vec
            projected_vectors.append(vec)
            total_energy += np.sum(vec)

        # 2. Resonance Focusing (Sigmoid)
        if len(projected_vectors) < 2:
            coherence = 1.0
            resonance_mask = np.ones(384)
        else:
            stack = np.stack(projected_vectors)
            variance = np.var(stack, axis=0)
            
            sensitivity = 20.0
            threshold = 0.05
            focus_signal = 1.0 / (1.0 + np.exp(sensitivity * (variance - threshold)))
            resonance_mask = focus_signal
                
            self.global_waveform *= resonance_mask
            coherence = float(np.mean(resonance_mask))


        # [PHASE 67] Dynamic Identity Check (Nature Clash)
        # Check if 'Physical Dynamics' (Qualia) are compatible.
        
        dynamic_penalty = 0.0
        if len(rotors) >= 2:
            count = 0
            for i in range(len(rotors)):
                for j in range(i + 1, len(rotors)):
                    d1 = rotors[i].dynamics
                    d2 = rotors[j].dynamics
                    
                    if d1 and d2:
                        count += 1
                        # Conflict 1: Chaos vs Order (Temperature Gap)
                        temp_diff = abs(d1.temperature - d2.temperature)
                        
                        # Conflict 2: Structure Clash
                        structure_conflict = 0.0
                        if (d1.rigidity > 0.8 and d2.fluidity > 0.8) or (d2.rigidity > 0.8 and d1.fluidity > 0.8):
                            structure_conflict = 1.0
                            
                        # Calculate Penalty
                        penalty = (temp_diff * 0.2) + (structure_conflict * 0.5)
                        dynamic_penalty += penalty
            
            if count > 0:
                dynamic_penalty /= count
            
        # Apply Penalty
        coherence = max(0.0, coherence - dynamic_penalty)


        # [PHASE 68] Somatic Impact (Phenomenology)
        # Meaning = Impact on the Body (Pain/Pleasure)
        total_pain = 0.0
        total_pleasure = 0.0
        
        if body:
            for r in rotors:
                if r.dynamics:
                    sensation = body.feel(r.dynamics)
                    if sensation.type == "PAIN":
                        total_pain += sensation.intensity
                    elif sensation.type == "PLEASURE":
                        total_pleasure += sensation.intensity
                        
            # Modulate Coherence based on Survival
            # Pain destroys coherence (Immediate Dissonance)
            if total_pain > 0:
                # If it hurts, it doesn't matter if the vectors align. It is WRONG.
                coherence = max(0.0, coherence - (total_pain * 2.0))
            
            # Pleasure reinforces coherence (Resonance)
            if total_pleasure > 0:
                coherence = min(1.0, coherence + (total_pleasure * 0.5))

        # 3. Calculate Entropy
        prob_dist = np.abs(self.global_waveform)
        if np.sum(prob_dist) > 0:
            prob_dist /= np.sum(prob_dist)
            entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-9))
        else:
            entropy = 0.0

        # Dominant Frequency
        dom_bin = np.argmax(self.global_waveform)
        dom_freq = 100.0 + (dom_bin / 384.0) * 1900.0

        state = ManifoldState(
            entropy=float(entropy),
            energy=float(total_energy),
            dominant_freq=float(dom_freq),
            coherence=float(coherence),
            pain=float(total_pain),
            pleasure=float(total_pleasure)
        )
        
        # 4. Generate Feedback
        feedback = self._generate_feedback(rotors, state, projected_vectors)
        
        return state, feedback

    def _generate_feedback(self, rotors: List[Rotor], state: ManifoldState, vectors: List[np.ndarray]) -> List[FeedbackSignal]:
        """Decides how to tune the rotors based on the global state."""
        signals = []
        
        # If Coherence is too low (Chaos), we need to create order.
        if state.coherence < 0.3:
            if np.sum(np.abs(self.global_waveform)) > 0:
                global_dir = self.global_waveform / np.linalg.norm(self.global_waveform)
                
                for i, r in enumerate(rotors):
                    if i >= len(vectors): break
                    v = vectors[i]
                    if np.linalg.norm(v) == 0: continue
                    v_dir = v / np.linalg.norm(v)
                    
                    alignment = np.dot(v_dir, global_dir)
                    
                    if alignment < 0.2:
                        signals.append(FeedbackSignal(r.name, "SHIFT_PHASE", np.pi/4))
                        signals.append(FeedbackSignal(r.name, "DAMPEN", 0.9))
                        
        return signals
