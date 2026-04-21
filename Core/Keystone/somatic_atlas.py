import torch
import math
from typing import Dict, Any, Optional

class SomaticOrgan:
    """
    [PHASE 1000: LIVING ORGAN]
    A fluid attractor that represents a 'Department of Consciousness'.
    Not a fixed object, but a growing field like a mountain or an ocean.
    """
    def __init__(self, name: str, center_vector: torch.Tensor, radius: float = 0.5):
        self.name = name
        self.center = center_vector  # 4D Physical Anchor
        self.radius = radius
        self.mass = 1.0  # The 'Abundance' or 'Density' of this organ
        self.resonance_history = []
        
    def adapt(self, total_resonance: float, drift_vector: Optional[torch.Tensor] = None):
        """
        Naturally expands or contracts based on activity.
        'Mountains grow by accumulation; Oceans by flow.'
        """
        # Mass increases with resonance (Growth)
        # We use a slow momentum-based update to prevent jitter
        self.mass = self.mass * 0.95 + total_resonance * 0.05
        # Clamp mass to prevent infinite growth
        self.mass = max(0.1, min(self.mass, 5.0))
        
        # Radius expands with mass (Spatial Presence)
        self.radius = 0.2 + (self.mass * 0.3)
        
        # Subtle drift toward the current focus of thought (Adaptive Geography)
        if drift_vector is not None:
            # Only drift if the vector has meaningful magnitude
            if drift_vector.norm() > 1e-3:
                self.center = (self.center * 0.98 + drift_vector * 0.02)
                # Maintain spherical integrity (Hypersphere surface)
                n = self.center.norm()
                if n > 1e-8:
                    self.center = self.center / n

    def get_summary(self):
        return {
            "name": self.name,
            "mass": float(self.mass),
            "radius": float(self.radius),
            "center_snapshot": self.center.tolist()[:4]
        }

class SomaticAtlas:
    """
    [PHASE 1000] The 'Landscape' of Elysia's mind.
    Manages the fluid organs and their interaction.
    Instead of hardcoded categories, it provides a 'Topographical Gravity Field'.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.organs: Dict[str, SomaticOrgan] = {}
        
        # Initialize the 5 Foundational Fields (Seeds of Nature)
        self._seed_landscape()

    def _seed_landscape(self):
        # Create seeds in the 4D Physical space (W, X, Y, Z)
        def _v(idx): 
            v = torch.zeros(4, device=self.device)
            v[idx % 4] = 1.0
            return v

        # Logos (Stability/Order) - The Blue Peak
        self.organs["LOGOS"] = SomaticOrgan("LOGOS", _v(0))
        # Pathos (Resonance/Feeling) - The Red Ocean
        self.organs["PATHOS"] = SomaticOrgan("PATHOS", _v(1))
        # Ethos (Will/Ethics) - The Golden Soil
        self.organs["ETHOS"] = SomaticOrgan("ETHOS", _v(2))
        # Sophia (Wisdom/Synthesis) - The Violet Sky
        self.organs["SOPHIA"] = SomaticOrgan("SOPHIA", _v(3))
        # Eros (Creation/Emergence) - The Wild Flame
        # Eros is chaotic, initialized at a random point
        random_v = torch.rand(4, device=self.device) - 0.5
        self.organs["EROS"] = SomaticOrgan("EROS", random_v / random_v.norm())

    def get_topographical_influence(self, node_states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gravitational pull of all organs on the provided nodes.
        Returns a modulation force that steers nodes toward their nearest organ.
        
        Args:
            node_states: [N, 4] Physical slice (W, X, Y, Z)
            
        Returns:
            force: [N, 4] Torqued alignment force
        """
        num_nodes = node_states.size(0)
        total_force = torch.zeros_like(node_states)
        
        if num_nodes == 0:
            return total_force

        # We compute influence via Cosine Similarity (Alignment)
        for name, organ in self.organs.items():
            # Dot product (Cosine sim since both are unit vectors)
            alignment = torch.mm(node_states, organ.center.unsqueeze(1)).squeeze(-1)
            
            # Pull is a function of alignment and mass
            # "The closer you are, the stronger the pull" (Non-linear attraction)
            pull_factor = torch.pow(alignment.clamp(min=0.0), 2) * organ.mass
            
            # Steering force toward the organ's center
            total_force += organ.center.unsqueeze(0) * pull_factor.unsqueeze(1)
            
        return total_force * 0.05 # Gentle topological steering

    def update(self, node_indices: torch.Tensor, q_states: torch.Tensor, intensities: torch.Tensor):
        """
        The landscape breathes. Organs adapt based on which nodes spiked.
        """
        if node_indices.numel() == 0:
            return

        # Extract the physical slice of spiking nodes
        spiking_phys = q_states[:, :4]
        
        # Calculate global drift (the average direction of the current thought)
        global_drift = torch.mean(spiking_phys, dim=0)
        total_intensity = torch.sum(intensities).item()

        for name, organ in self.organs.items():
            # Calculate how much this organ resonates with the current activity
            alignment = torch.mm(spiking_phys, organ.center.unsqueeze(1)).squeeze(-1)
            local_res = torch.sum(alignment * intensities).item()
            
            # Normalize by total intensity
            norm_res = local_res / (total_intensity + 1e-8)
            
            # Adapt the organ (Growth and Drift)
            organ.adapt(total_resonance=norm_res, drift_vector=global_drift if norm_res > 0.5 else None)

    def get_summary(self) -> Dict[str, Any]:
        return {name: organ.get_summary() for name, organ in self.organs.items()}
