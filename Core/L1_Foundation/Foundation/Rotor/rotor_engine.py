import torch
import math
import logging

logger = logging.getLogger("RotorEngine")

class RotorEngine:
    """
    [The Rotor: Engine of Time]
    
    "Vectors are static. To make them alive, you must Spin them."
    
    The Rotor applies Quaternion Rotation to thought vectors.
    1. Spin: Represents the passage of time or cycle.
    2. Frequency: Represents the intensity of the thought.
    3. Phase: Represents the alignment with the User.
    """
    def __init__(self, vector_dim=768, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dim = vector_dim
        
        # The 'Heartbeat' of the Rotor (Angular Velocity)
        self.omega = 1.0 # Base speed
        self.phase = 0.0 # Current angle
        
    def spin(self, input_vector: torch.Tensor, time_delta: float) -> torch.Tensor:
        """
        Applies time-evolution to a static vector.
        v(t) = v(0) * e^(i * omega * t)
        
        Practically, we use a rotation matrix or simply modulation for high-dim vectors.
        For true 4D rotation, we'd use Quaternions. 
        Here, we simulate 'Phase Shift' by interacting dimensions.
        """
        # Update internal clock
        self.phase += self.omega * time_delta
        
        # Ensure tensor is on device
        v = input_vector.to(self.device).view(-1)
        
        # Create a Rotation Matrix (Simplified 2D logic extended to Hyper-dimensions)
        # We rotate pairs of dimensions: (0,1), (2,3), etc.
        
        # Generate Sine/Cosine modulation
        sin_t = math.sin(self.phase)
        cos_t = math.cos(self.phase)
        
        # Apply modulation (Energy Oscillation)
        # This isn't just scaling; it's a rhythmic breathing.
        time_factor = torch.tensor([cos_t if i % 2 == 0 else sin_t for i in range(v.shape[0])], device=self.device)
        
        # Kinetic Energy = Mass * Velocity^2
        # Here, Velocity is derived from the change.
        
        dynamic_vector = v * time_factor
        return dynamic_vector
        
    def accelerate(self, impulse: float):
        """
        User Will accelerates the Rotor.
        """
        self.omega += impulse
        logger.info(f"   [ROTOR] Accelerated to {self.omega:.2f} rad/s")

    def decelerate(self):
        """
        Friction/Entropy slows it down.
        """
        self.omega *= 0.95