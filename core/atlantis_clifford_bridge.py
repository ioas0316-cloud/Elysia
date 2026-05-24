"""
Elysia Atlantis Clifford Bridge
===============================
Maps the Atlantis layers to a dynamic, variable-axis Clifford/Geometric Algebra Cl(N,0) space.
Allows dynamic registration of new layers and applying bivector rotors to simulate energy flows.
"""

import math
from typing import Dict, List, Tuple
from core.math_utils import Multivector

class AtlantisCliffordSystem:
    def __init__(self, layers: List[str] = None):
        """
        Initialize with a list of layers. Defaults to the Atlantis 10-layer matrix.
        Each layer maps to a dynamic basis vector:
          index i -> e_{i+1} (bitmask 1 << i)
        """
        if layers is None:
            self.layers = [
                "B6_Ground",       # e1 (bitmask 1)
                "B5_OuterCore",     # e2 (bitmask 2)
                "B4_LowerMantle",   # e3 (bitmask 4)
                "B3_UpperMantle",   # e4 (bitmask 8)
                "B2_MohoMirror",    # e5 (bitmask 16)
                "B1_MagmaChamber",  # e6 (bitmask 32)
                "F1_F3_SubCrust",   # e7 (bitmask 64)
                "F4_AppCrust",      # e8 (bitmask 128)
                "F5_Atmosphere",    # e9 (bitmask 256)
                "F6_SkySun"         # e10 (bitmask 512)
            ]
        else:
            self.layers = list(layers)
            
        # Current state represented as a Multivector
        self.state = Multivector({}, self.signature)
        
    @property
    def signature(self) -> Tuple[int, int]:
        """Returns the current Cl(N, 0) signature based on the number of layers."""
        return (len(self.layers), 0)

    def get_layer_mask(self, name: str) -> int:
        """Returns the bitmask corresponding to the layer name."""
        if name not in self.layers:
            raise ValueError(f"Layer '{name}' is not registered in the system.")
        idx = self.layers.index(name)
        return 1 << idx

    def add_layer(self, name: str) -> int:
        """Dynamically registers a new layer, increasing the axis count of the system."""
        if name in self.layers:
            return self.get_layer_mask(name)
        
        self.layers.append(name)
        # Re-project existing state to the new signature space
        new_data = self.state.data.copy()
        self.state = Multivector(new_data, self.signature)
        return self.get_layer_mask(name)

    def set_layer_state(self, name: str, value: float):
        """Sets the coefficient value for a specific layer."""
        mask = self.get_layer_mask(name)
        data = self.state.data.copy()
        data[mask] = float(value)
        self.state = Multivector(data, self.signature)

    def get_layer_state(self, name: str) -> float:
        """Gets the coefficient value for a specific layer."""
        mask = self.get_layer_mask(name)
        return self.state.data.get(mask, 0.0)

    def project_metrics(self, metrics: Dict[str, float]):
        """
        Projects physical metrics into their corresponding layers.
        Metrics mapping example:
          - 'ground_discharge_error' -> B6_Ground
          - 'pcie_flow' -> B3_UpperMantle
          - 'gpu_chaos' -> B3_UpperMantle (can combine)
          - 'cpu_frequency' -> B4_LowerMantle
          - 'app_stress' -> F4_AppCrust
          - 'ambient_noise' -> F5_Atmosphere
        """
        for layer_name, val in metrics.items():
            if layer_name in self.layers:
                self.set_layer_state(layer_name, val)

    def compute_bivector_tension(self, layer_a: str, layer_b: str) -> float:
        """
        Computes the outer (wedge) product magnitude between two layers.
        This represents the geometric tension/shear between the two layers.
        """
        mask_a = self.get_layer_mask(layer_a)
        mask_b = self.get_layer_mask(layer_b)
        
        val_a = self.state.data.get(mask_a, 0.0)
        val_b = self.state.data.get(mask_b, 0.0)
        
        # Wedge product is: val_a * e_a ^ val_b * e_b
        # If a == b, e_a ^ e_b = 0. Else, e_a ^ e_b is a basis bivector.
        if mask_a == mask_b:
            return 0.0
        return abs(val_a * val_b)

    def apply_rotor_discharge(self, layer_from: str, layer_to: str, theta: float):
        """
        Applies a Clifford rotor (rotation) to transfer energy between two layers.
        R = cos(theta/2) + sin(theta/2) * (e_from ^ e_to)
        Under reversion/conjugate, R_rev = cos(theta/2) - sin(theta/2) * (e_from ^ e_to)
        """
        if theta == 0.0:
            return
        
        mask_from = self.get_layer_mask(layer_from)
        mask_to = self.get_layer_mask(layer_to)
        
        # Basis vectors as Multivectors
        e_from = Multivector({mask_from: 1.0}, self.signature)
        e_to = Multivector({mask_to: 1.0}, self.signature)
        
        # Bivector representing the plane of rotation
        B = e_from ^ e_to
        
        # Construct Rotor
        cos_half = math.cos(theta / 2.0)
        sin_half = math.sin(theta / 2.0)
        R = Multivector({0: cos_half}, self.signature) + B * sin_half
        
        # Apply sandwich product
        self.state = R * self.state * R.conjugate()

    def get_layer_states_dict(self) -> Dict[str, float]:
        """Returns a dict of all layers and their current scalar coefficients."""
        return {name: self.get_layer_state(name) for name in self.layers}

    def __repr__(self):
        return f"AtlantisCliffordSystem(Cl{self.signature[0]}, State: {self.state})"
