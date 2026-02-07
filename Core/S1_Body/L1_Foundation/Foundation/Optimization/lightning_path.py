try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None
import time


class PsychField:
    """
    [Field-Based Psychodynamics]
    Maps individual consciousness structures (Enneagram/MBTI) to Spatial Fields.
    Instead of simulating an internal brain for every NPC, we create a
    'Meaning Field' that pulls them towards their natural niche.
    
    Layers:
    1. Body (Gut/Instinct): Preference for Activity/Environment.
    2. Mind (Head/Thinking): Preference for Information/Logic.
    3. Spirit (Heart/Feeling): Preference for Value/Connection.
    """
    @staticmethod
    def generate_layers(shape, rotors: dict, device: str = 'cpu'):
        """
        Generates 3-Layer Tensor Field based on Rotor 'Will'.
        """
        device = torch.device(device)
        
        # 1. Extract Archetypal Forces from Rotors
        # Example: 'Purpose' rotor drives the Spirit Layer (Value)
        # 'FluxLight' drives the Body Layer (Activity)
        
        force_body = rotors.get("FluxLight", 0.5)   # Activity/Change
        force_mind = rotors.get("HyperCosmos", 0.5) # Complexity/Logic
        force_spirit = rotors.get("Identity", 0.5)  # Meaning/Self
        
        # 2. Generate Base Fields (Gradients)
        x = torch.linspace(-1, 1, shape[0], device=device)
        y = torch.linspace(-1, 1, shape[1], device=device)
        xv, yv = torch.meshgrid(x, y, indexing='ij')
        
        # Body Field: High Activity vs Stability
        # Pattern: Turbulence (Simulated by high freq sine)
        body_field = force_body * torch.sin(xv * 10.0 + yv * 10.0)
        
        # Mind Field: High Information vs Simplicity
        # Pattern: Concentric circles (Centralized knowledge vs Distributed)
        r = torch.sqrt(xv**2 + yv**2)
        mind_field = force_mind * torch.exp(-r * 2.0)
        
        # Spirit Field: Connection vs Autonomy
        # Pattern: Directional Gradient (North = Autonomy, South = Connection)
        spirit_field = force_spirit * yv
        
        return torch.stack([body_field, mind_field, spirit_field])

class LightningPath:
    """
    [Hardware Sovereignty]
    Optimized Data Bridge between L6 (Governance/Rotor) and L4 (World Field).
    Uses Vectorized Operations (Numpy/Torch) instead of loops.
    """
    
    def __init__(self, world_shape, device='cpu'):
        self.device = device
        self.shape = world_shape
        # The 'Field' is a tensor representing the potential energy of every point in space
        self.field_tensor = torch.zeros(world_shape, device=self.device)
        self.psych_tensor = torch.zeros((3, *world_shape), device=self.device) # [3, H, W]
        self.last_pulse_time = 0
        
    def project_will(self, rotor_energies: dict, field_channel: str = "will"):
        """
        Projects the 'Will' of the Rotors onto the Fabric of Reality.
        """
        # 1. Update Physical Will Field (Existing Logic)
        total_will = sum(rotor_energies.values())
        x = torch.linspace(-1, 1, self.shape[0], device=self.device)
        y = torch.linspace(-1, 1, self.shape[1], device=self.device)
        xv, yv = torch.meshgrid(x, y, indexing='ij')
        radius = torch.sqrt(xv**2 + yv**2)
        frequency = 10.0 * total_will
        phase = time.time() * frequency
        wave_pattern = total_will * torch.sin(radius * frequency - phase)
        self.field_tensor = wave_pattern
        
        # 2. [NEW] Update Psych Fields (Body, Mind, Spirit)
        self.psych_tensor = PsychField.generate_layers(self.shape, rotor_energies, device=self.device)
        
        return self.field_tensor

    def get_psych_snapshot(self):
        return self.psych_tensor.cpu().numpy()

    def get_field_snapshot(self):
        return self.field_tensor.cpu().numpy()

    def snatch_feedback(self, world_state_tensor) -> float:
        if not isinstance(world_state_tensor, torch.Tensor):
            world_state_tensor = torch.tensor(world_state_tensor, device=self.device)
        coherence = torch.mean(self.field_tensor * world_state_tensor)
        return coherence.item()
