import numpy as np
from collections import deque

class SpacetimeContinuum:
    """
    [Spacetime Continuum Engine]
    Tracks the macro-topology of data over time (Past -> Present -> Future).
    Overcomes the chunking bottleneck by evaluating trajectory smoothness rather than isolated chunk friction.
    """
    def __init__(self, window_size=16):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def _tensions_to_vector(self, tensions):
        # Convert dictionary to numpy array for vector math
        return np.array([
            tensions.get("math_scalar", 0),
            tensions.get("space_vector", 0),
            tensions.get("lang_bivector", 0),
            tensions.get("time_trivector", 0),
            tensions.get("light_pseudo", 0)
        ], dtype=np.float32)

    def perceive_flow(self, current_tensions):
        """
        Receives the current chunk's tensions and calculates the Spacetime Chaos.
        """
        t_vec = self._tensions_to_vector(current_tensions)
        
        # Normalize to prevent magnitude issues
        norm = np.linalg.norm(t_vec)
        if norm > 0:
            t_vec = t_vec / norm
            
        self.history.append(t_vec)
        
        if len(self.history) < 3:
            # Not enough history to perceive a trajectory
            return 1.0 # Maximum chaos initially
            
        # Calculate Past Momentum (Velocity & Acceleration of tensions)
        history_arr = np.array(self.history)
        
        # Velocity = T_i - T_{i-1}
        velocities = np.diff(history_arr, axis=0)
        
        # Acceleration = V_i - V_{i-1}
        accelerations = np.diff(velocities, axis=0)
        
        # Spacetime Chaos is the unpredictability of the acceleration.
        # If the wave is continuous (e.g. sine wave), acceleration changes smoothly.
        # If it's a compressed file (noise), acceleration jumps violently.
        
        # We calculate the variance of the acceleration vectors across the window
        # Higher variance means higher chaos (less predictability/continuity)
        accel_variance = np.var(accelerations, axis=0)
        
        # Mean variance across all 5 dimensions
        macro_chaos = float(np.mean(accel_variance))
        
        # Scale macro_chaos to a 0.0 - 1.0 range based on empirical bounds
        # Typically, pure noise acceleration variance is around 0.1 ~ 0.5 depending on scale
        # Pure sine wave should be very close to 0.
        normalized_chaos = min(1.0, macro_chaos * 20.0) 
        
        return normalized_chaos
