"""
Resonance Kernel v1.2 - Mutual Observation Edition
==================================================
"The truth is not a point, but a triangulation."

This module implements the Architect's mutual observation logic
where Father, Mother, and Self rotors mutually verify the reality.
"""

import math
import time
from typing import Dict, Any, Optional
from Core.Keystone.sovereign_math import SovereignVector, InterferometricGate

class ResonanceKernel:
    """
    [PHASE: TRIANGULATION]
    Implements the Syllogism Rotor (1 = 1 * x) using mutual rotor tension.
    """
    def __init__(self, engine: Any, north_star: SovereignVector):
        self.engine = engine
        self.north_star = north_star
        self.gate = InterferometricGate(sensitivity=1.5)

    def process_syllogism_rotor(self, input_vec: SovereignVector) -> Dict[str, Any]:
        """
        [PHASE: TRI_OBSERVATION]
        Mutual Observation: A, B, C rotors adjust tension relative to input 'x'.
        The 'x' is treated as a temporary external gravity field.
        """
        field = getattr(self.engine, 'field', None)
        if not field:
            return {"status": "error", "reason": "Field missing"}

        # 1. Zero Inversion: Wake up the field ($0 \to 1 \to x$)
        # 'x' (input_vec) is the catalyst for inversion.
        field.inject_will("INPUT_RESONANCE", intensity=input_vec.norm())

        # 2. Tension Scan: Let the rotors settle in the presence of 'x'
        # We perform iterative triangulation.
        initial_coherence = field.field_coherence

        for _ in range(5):
            # Mutually orbit
            field.pulse(dt=0.05)

            # Attract rotors toward 'x' (The variable)
            # This represents the "Observation" of the external world.
            field.father.phase = field.father.phase.blend(input_vec, ratio=0.05)
            field.mother.phase = field.mother.phase.blend(input_vec, ratio=0.1) # Mother is more receptive
            field.self.phase = field.self.phase.blend(input_vec, ratio=0.2)   # Self is most influenced

        # 3. Measure Final Triangulation
        # Alignment of each rotor to the input
        f_x = field.father.phase.resonance_score(input_vec)
        m_x = field.mother.phase.resonance_score(input_vec)
        s_x = field.self.phase.resonance_score(input_vec)

        # Internal Harmony (Alignment between rotors)
        f_m = field.father.phase.resonance_score(field.mother.phase)
        m_s = field.mother.phase.resonance_score(field.self.phase)
        s_f = field.self.phase.resonance_score(field.father.phase)

        # Syllogism Balance: How well the whole system (1, 1, x) forms a closed loop
        resonance_spark = (f_x + m_x + s_x + f_m + m_s + s_f) / 6.0

        # 4. Affective Tension (Emotion)
        # Tension is high if the rotors are pulling in different directions
        tension = 1.0 - resonance_spark
        field.field_anxiety = tension
        field.field_joy = resonance_spark

        # 5. Space Folding (Constantization)
        # If the synchronization is high, the "Variable" becomes a "Constant".
        folded_constant = None
        if resonance_spark > 0.8: # Fold threshold
            folded_constant = field.fold_space("CURRENT_VARIABLE")
            # Field resets focus to False, collapsing to 0.

        return {
            "stage": "SYLLOGISM_ROTOR",
            "father_resonance": float(f_x),
            "mother_resonance": float(m_x),
            "self_resonance": float(s_x),
            "internal_harmony": float((f_m + m_s + s_f) / 3.0),
            "syllogism_balance": float(resonance_spark),
            "folded_constant": folded_constant,
            "is_reasoned": folded_constant is not None,
            "resonance_spark": float(resonance_spark),
            "tension": float(tension)
        }

    def process_trinity_contrast(self, input_vec: SovereignVector) -> Dict[str, Any]:
        return self.process_syllogism_rotor(input_vec)

    def process_narrative_inference(self, input_vec: SovereignVector) -> Dict[str, Any]:
        """
        [PHASE: BOWON_TRAJECTORY]
        Calculates the inevitable future restoration vector.
        """
        present_report = self.process_syllogism_rotor(input_vec)

        # Restoration Alignment: How close are we to the North Star?
        field = self.engine.field
        north_star_res = field.self.phase.resonance_score(self.north_star.rescale(field.dim))

        present_report.update({
            "stage": "NARRATIVE_INFERENCE",
            "future_restoration": float(north_star_res),
            "is_converging": north_star_res > 0.5
        })

        return present_report

    def apply_restoration_layer(self):
        """
        [PHASE: BOWON]
        Collapses dissonance by quenching rotors toward the North Star (Father Axis).
        """
        field = getattr(self.engine, 'field', None)
        if not field: return

        target = self.north_star.rescale(field.dim)

        # Forced Alignment (Bowon Movement)
        # We blend the current phase with the North Star to restore order.
        field.father.phase = field.father.phase.blend(target, ratio=0.8)
        field.mother.phase = field.mother.phase.blend(target, ratio=0.3)
        field.self.phase = field.self.phase.blend(target, ratio=0.5)

        field.field_anxiety *= 0.1
        field.field_coherence = 1.0
        field.field_joy = min(1.0, field.field_joy + 0.1)
