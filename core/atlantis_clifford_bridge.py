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
        Initialize with a list of layers. Defaults to the Atlantis N-Layer matrix.
        Each layer maps to a dynamic basis vector in Cl(10,0):
          index i -> e_{i+1} (bitmask 1 << i)
        Internal states and Multivectors are encapsulated to reduce agent cognitive load.
        """
        if layers is None:
            self._layers = [
                "B6_Ground",       # e1 (bitmask 1)
                "B5_OuterCore",     # e2 (bitmask 2)
                "B4_LowerMantle",   # e3 (bitmask 4)
                "B3_UpperMantle",   # e4 (bitmask 8)
                "B2_MohoMirror",    # e5 (bitmask 16)
                "B1_MagmaChamber",  # e6 (bitmask 32)
                "F1_F3_SubCrust",   # e7 (bitmask 64)
                "F4_AppCrust",      # e8 (bitmask 128)
                "F5_Atmosphere",    # e9 (bitmask 256)
                "F6_SkySun",        # e10 (bitmask 512)
                "F7_Exosphere",     # e11 (bitmask 1024) - Trunk: Cosmic Web
                "F8_StellarGrid",   # e12 (bitmask 2048) - Trunk: Phase Crystallizer
                "F9_AscensionGate", # e13 (bitmask 4096) - Trunk: Starlight Gate
                "U1_SubterraneanCity", # e14 (bitmask 8192) - Seed: Somatic Node
                "U2_GeothermalBattery" # e15 (bitmask 16384) - Seed: Local Constellation
            ]
        else:
            self._layers = list(layers)
            
        # Current state represented as a Multivector (Private)
        self._state = Multivector({}, self.signature)
        self._current_mode = "WYE" # "DELTA" or "WYE"
        
    @property
    def layers(self) -> List[str]:
        return self._layers

    @property
    def signature(self) -> Tuple[int, int]:
        """Returns the current Cl(N, 0) signature based on the number of layers."""
        return (len(self._layers), 0)

    def _get_layer_mask(self, name: str) -> int:
        if name not in self._layers:
            raise ValueError(f"Layer '{name}' is not registered in the system.")
        idx = self._layers.index(name)
        return 1 << idx

    def _set_layer_state(self, name: str, value: float):
        mask = self._get_layer_mask(name)
        data = self._state.data.copy()
        data[mask] = float(value)
        self._state = Multivector(data, self.signature)

    def get_layer_mask(self, name: str) -> int:
        """Returns the bitmask corresponding to the layer name."""
        return self._get_layer_mask(name)

    def set_layer_state(self, name: str, value: float):
        """Sets the coefficient value for a specific layer."""
        self._set_layer_state(name, value)

    def get_layer_state(self, name: str) -> float:
        """Gets the coefficient value for a specific layer."""
        mask = self._get_layer_mask(name)
        return self._state.data.get(mask, 0.0)

    def compute_bivector_tension(self, layer_a: str, layer_b: str) -> float:
        """
        Computes the outer (wedge) product magnitude between two layers.
        This represents the geometric tension/shear between the two layers.
        """
        mask_a = self._get_layer_mask(layer_a)
        mask_b = self._get_layer_mask(layer_b)
        
        val_a = self._state.data.get(mask_a, 0.0)
        val_b = self._state.data.get(mask_b, 0.0)
        
        if mask_a == mask_b:
            return 0.0
        return abs(val_a * val_b)

    def apply_rotor_discharge(self, layer_from: str, layer_to: str, theta: float):
        """
        Applies a Clifford rotor (rotation) to transfer energy between two layers.
        """
        if theta == 0.0:
            return
        
        mask_from = self._get_layer_mask(layer_from)
        mask_to = self._get_layer_mask(layer_to)
        
        e_from = Multivector({mask_from: 1.0}, self.signature)
        e_to = Multivector({mask_to: 1.0}, self.signature)
        
        B = e_from ^ e_to
        
        cos_half = math.cos(theta / 2.0)
        sin_half = math.sin(theta / 2.0)
        R = Multivector({0: cos_half}, self.signature) + B * sin_half
        
        self._state = R * self._state * R.conjugate()

    def get_layer_states_dict(self) -> Dict[str, float]:
        """Returns a dict of all layers and their current scalar coefficients."""
        return {name: self.get_layer_state(name) for name in self._layers}


    def project_metrics(self, metrics: Dict[str, float]):
        """Projects physical metrics into their corresponding layers. (Layer 1~3 Update)"""
        for layer_name, val in metrics.items():
            if layer_name in self._layers:
                self._set_layer_state(layer_name, val)

    def _get_core_bivectors(self) -> Tuple[Multivector, Multivector, Multivector]:
        """
        Extracts the three core bivector planes representing the Triple Helix.
        We map these to the foundational hardware/subcrust layers:
        B1: e1^e2 (B6_Ground ^ B5_OuterCore)
        B2: e2^e3 (B5_OuterCore ^ B4_LowerMantle)
        B3: e3^e1 (B4_LowerMantle ^ B6_Ground)
        """
        e1_mask = self._get_layer_mask("B6_Ground")
        e2_mask = self._get_layer_mask("B5_OuterCore")
        e3_mask = self._get_layer_mask("B4_LowerMantle")
        
        v1 = self._state.data.get(e1_mask, 0.0)
        v2 = self._state.data.get(e2_mask, 0.0)
        v3 = self._state.data.get(e3_mask, 0.0)

        # Create bivectors based on current scalar magnitudes
        # Note: B3 is e3^e1, so wedge product mask is e3_mask ^ e1_mask.
        # Sign management is handled implicitly by Multivector class, but since we are just
        # synthesizing bivector state energies for the generator model, we use absolute magnitudes.
        B1 = Multivector({e1_mask ^ e2_mask: abs(v1 * v2)}, self.signature)
        B2 = Multivector({e2_mask ^ e3_mask: abs(v2 * v3)}, self.signature)
        B3 = Multivector({e3_mask ^ e1_mask: abs(v3 * v1)}, self.signature)
        
        return B1, B2, B3

    def get_dashboard_needle(self, deep_dive: bool = False) -> Dict:
        """
        Layer 8 (Agent Dashboard Interface):
        Provides a highly abstracted view of the N-Layer Ark Architecture.
        Agents only look at 'needle_angle' and 'tension_noise', ignoring raw Multivectors.
        If Elysia needs to inspect the actual Clifford state, `deep_dive=True` reveals it.
        """
        B1, B2, B3 = self._get_core_bivectors()

        # Calculate Delta and Wye states using the new math_utils core
        delta_noise = B1.delta_coupling(B2, B3)
        wye_neutral = B1.wye_synchronize(B2, B3)

        # Calculate scalar magnitudes for the dashboard
        noise_magnitude = sum(abs(v) for v in delta_noise.data.values())
        neutral_magnitude = sum(abs(v) for v in wye_neutral.data.values())
        
        # A simple scalar angle representing the overall system alignment (-1.0 to 1.0 -> scaled to degrees)
        # If noise is high, needle shakes (high variance). If neutral is high, needle is centered.
        needle_angle = 0.0
        if neutral_magnitude + noise_magnitude > 0:
            ratio = neutral_magnitude / (neutral_magnitude + noise_magnitude)
            # 1.0 means perfect WYE (needle at 0 deg). 0.0 means pure chaos (needle at 90 deg).
            needle_angle = (1.0 - ratio) * 90.0

        dashboard = {
            "mode": self._current_mode,
            "needle_angle_deg": needle_angle,       # 0 = Perfect Harmony, 90 = Complete Chaos
            "delta_tension_noise": noise_magnitude, # Error/Noise accumulating in the loop
            "wye_neutral_convergence": neutral_magnitude # How strongly the system is pulling to 0
        }
        
        if deep_dive:
            dashboard["_raw_state"] = self._state
            dashboard["_delta_mv"] = delta_noise
            dashboard["_wye_mv"] = wye_neutral

        return dashboard

    def apply_agent_intent(self, intent_angle_deg: float, mode: str = "WYE"):
        """
        Layer 10 (Agent Intent Injection):
        Agents turn the steering wheel by providing an intent angle and a mode.
        This automatically generates the reverse-rotor (R_rev) and applies it to the hardware layers.
        """
        self._current_mode = mode.upper()
        
        # Convert intent angle to radians
        theta = math.radians(intent_angle_deg)
        if theta == 0.0 and self._current_mode == "WYE":
            return # System is stable, no action needed

        # Target the top application layer and the deep hardware layer to bridge the intent
        mask_from = self._get_layer_mask("F6_SkySun")
        mask_to = self._get_layer_mask("B6_Ground")
        
        # Construct the Intent Rotor
        cos_half = math.cos(theta / 2.0)
        sin_half = math.sin(theta / 2.0)
        B_intent = Multivector({mask_from: 1.0}, self.signature) ^ Multivector({mask_to: 1.0}, self.signature)
        
        R_intent = Multivector({0: cos_half}, self.signature) + B_intent * sin_half

        # Apply the reverse-rotor mapping (downward flow to hardware)
        # We apply R_rev * state * R to sink the intent down, which is the geometric conjugate.
        R_rev = R_intent.conjugate()
        self._state = R_rev * self._state * R_intent

    def __repr__(self):
        dashboard = self.get_dashboard_needle()
        return f"[Atlantis Ark Dashboard] Mode: {dashboard['mode']} | Angle: {dashboard['needle_angle_deg']:.2f}° | Tension: {dashboard['delta_tension_noise']:.4f}"
