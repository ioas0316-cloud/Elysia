import numpy as np
from pulse import EnergyPulse, CircuitException
from rotor_math import FractalRotorMath

class BaseNode:
    def __init__(self, node_id):
        self.id = node_id
        self.internal_phase = np.random.uniform(0, 2*np.pi)
        self.base_type = "Generic"

    def process(self, pulse):
        raise NotImplementedError

class Base2Node(BaseNode):
    """
    2-Base Circuit (Binary/Linear): Rigid toggle.
    Can only handle discrete, highly aligned phases.
    """
    def __init__(self, node_id):
        super().__init__(node_id)
        self.base_type = 2
        # Internal state is either 0 or pi
        self.internal_phase = np.random.choice([0.0, np.pi])

    def process(self, pulse):
        phase_diff = np.abs(self.internal_phase - pulse.phase) % (2*np.pi)

        # If phase isn't close to 0 or pi, the binary circuit shatters
        if 0.5 < phase_diff < (np.pi - 0.5) or (np.pi + 0.5) < phase_diff < (2*np.pi - 0.5):
            raise CircuitException(
                base_type=self.base_type,
                node_id=self.id,
                phase_mismatch=phase_diff,
                frequency=pulse.frequency * 2.5 # high frequency spike on failure
            )

        # Snap to binary state
        new_phase = 0.0 if phase_diff < np.pi/2 else np.pi
        self.internal_phase = new_phase

        return EnergyPulse(pulse.amplitude, pulse.frequency, self.internal_phase)

class Base3Node(BaseNode):
    """
    3-Base Circuit (Ternary): Amplification (+), Neutral (0), Suppression (-).
    Allows for more variance, filters energy based on resonance.
    """
    def __init__(self, node_id):
        super().__init__(node_id)
        self.base_type = 3

    def process(self, pulse):
        phase_diff = np.abs(self.internal_phase - pulse.phase) % (2*np.pi)

        # Determine state: Amplified (close to internal), Suppressed (opposite), Neutral (orthogonal)
        if phase_diff < np.pi/3:
            # Amplify
            new_amp = pulse.amplitude * 1.5
            state = "Amplify"
        elif phase_diff > 2*np.pi/3:
            # Suppress
            new_amp = pulse.amplitude * 0.5
            state = "Suppress"
        else:
            # Neutral / Dissipate
            new_amp = pulse.amplitude * 0.9
            state = "Neutral"

        # Circuit fails if amplitude drops too low (energy death) or goes too high (meltdown)
        if new_amp < 0.1 or new_amp > 100.0:
            raise CircuitException(
                base_type=self.base_type,
                node_id=self.id,
                phase_mismatch=phase_diff,
                frequency=pulse.frequency
            )

        self.internal_phase = (self.internal_phase + pulse.phase) / 2.0 # align slightly to pulse

        return EnergyPulse(new_amp, pulse.frequency, self.internal_phase, origin_text=f"Base3({state})")

class Base4Node(BaseNode):
    """
    4-Base Circuit (Quaternion/4D Tensor): Handles multi-dimensional rotational shift.
    Uses FractalRotorMath to perform quaternion rotations.
    """
    def __init__(self, node_id):
        super().__init__(node_id)
        self.base_type = 4
        self.math = FractalRotorMath()
        self.internal_q = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z

    def process(self, pulse):
        # We need text to map to a quaternion, but pulse only has physical properties.
        # We will use pulse.origin_text if available, else derive a quaternion from properties.

        if pulse.origin_text and pulse.origin_text != "":
            q_pulse = self.math.text_to_quaternion(pulse.origin_text)
        else:
            # Fallback: create a quaternion representing rotation around Z by pulse.phase
            q_pulse = np.array([np.cos(pulse.phase/2), 0.0, 0.0, np.sin(pulse.phase/2)])

        theta = self.math.quaternion_angle(self.internal_q, q_pulse)

        if theta > 1.5: # Critical rotational mismatch threshold
             raise CircuitException(
                base_type=self.base_type,
                node_id=self.id,
                phase_mismatch=theta,
                frequency=pulse.frequency * theta
            )

        # Hamilton product to combine internal rotation with pulse rotation
        self.internal_q = self.math.quaternion_multiply(self.internal_q, q_pulse)

        # The new phase is derived from the scalar part (w) of the new quaternion
        new_phase = np.arccos(self.internal_q[0]) * 2

        return EnergyPulse(pulse.amplitude, pulse.frequency + (theta*10), new_phase, origin_text="Base4(QuantumShift)")
