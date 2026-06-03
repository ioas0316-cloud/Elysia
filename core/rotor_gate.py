import math
from typing import Dict, List, Tuple

class ConceptWave:
    def __init__(self, name: str):
        self.name = name
        # Dictionary of Axis Name -> Phase Angle (in radians)
        self.phases: Dict[str, float] = {}
        # Dynamic velocity vector: determines the "movement" or "seeking" direction
        # Mapping Axis Name -> Momentum/Velocity
        self.velocity: Dict[str, float] = {}
        
    def add_axis(self, axis: str, phase: float = 0.0):
        if axis not in self.phases:
            self.phases[axis] = phase
            self.velocity[axis] = 0.0

    def get_phase(self, axis: str) -> float:
        return self.phases.get(axis, None)

    def apply_force(self, axis: str, force: float):
        """ Applies a kinetic force, updating the velocity of the concept. """
        if axis in self.velocity:
            self.velocity[axis] += force
        else:
            self.velocity[axis] = force

    def __repr__(self):
        phase_str = ", ".join([f"{k}: {math.degrees(v):.0f}°" for k, v in self.phases.items()])
        vel_str = ", ".join([f"{k}: {v:.2f}" for k, v in self.velocity.items() if v != 0])
        return f"ConceptWave({self.name} | Phases: [{phase_str}] | Vel: [{vel_str}])"

class RotorGate:
    """
    RotorGate acts not as a static 'if-else' judgment, but as an interaction 
    mechanism that transfers energy, alters phases, and induces kinetic movement.
    """
    def __init__(self):
        pass

    def interact(self, a: ConceptWave, b: ConceptWave, criteria_axis: str) -> str:
        """
        Calculates the interaction between two concepts based on a criteria axis.
        Returns a string describing the resulting dynamic state.
        """
        phase_a = a.get_phase(criteria_axis)
        phase_b = b.get_phase(criteria_axis)

        # If neither concept has the axis, they cannot interact on it
        if phase_a is None or phase_b is None:
            return f"No interaction on {criteria_axis} (Axis not found in one or both)."

        phase_diff = abs(phase_a - phase_b)

        # 1. Similarity / Alignment Principle (같음의 공리)
        if math.isclose(phase_diff, 0.0, abs_tol=1e-5):
            # Instead of a static "True", generate kinetic momentum (방향성/운동성)
            # The resonance creates a forward thrust on this axis, making them "seek" further connections.
            resonance_force = 1.0 
            a.apply_force(criteria_axis, resonance_force)
            b.apply_force(criteria_axis, resonance_force)
            return f"[{a.name}] & [{b.name}] resonated on '{criteria_axis}'. Gained kinetic velocity. (Attraction/Movement)"

        # 2. Differentiation / Phase Shift Principle (다름의 공리)
        else:
            # They share the axis but have a phase difference. 
            # This generates a structural Tension (관계성) rather than a simple 'False'.
            tension = math.sin(phase_diff)
            # This tension could act as a centripetal force holding them in orbit
            return f"[{a.name}] & [{b.name}] have a phase difference of {math.degrees(phase_diff):.0f}° on '{criteria_axis}'. Tension: {tension:.2f}. (Structural Bond)"

    def differentiate_new_axis(self, a: ConceptWave, b: ConceptWave, base_axis: str, new_axis: str):
        """
        When two aligned concepts need to be differentiated by a new perspective,
        the RotorGate dynamically splits the dimension, shifting one's phase orthogonally.
        """
        # Assume a stays at 0 degrees on the new axis, b shifts to 90 degrees (pi/2)
        a.add_axis(new_axis, phase=0.0)
        b.add_axis(new_axis, phase=math.pi / 2)
        return f"Axis split occurred! '{new_axis}' created. [{b.name}] shifted by 90° relative to [{a.name}]."
