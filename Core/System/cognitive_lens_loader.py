import os
import json
import math
from Core.Keystone.sovereign_math import SovereignVector

class CognitiveLensLoader:
    """
    [Phase 1260: Fluid Cognitive Lens Loader]
    Loads crystallized model files (e.g. gpt2, phi-3 output json) 
    and transforms their multi-rotor phase structures into active cognitive filters.
    Supports dynamic dimensional scaling and variable active lenses.
    """
    def __init__(self, crystal_path=None):
        if crystal_path is None:
            # Fallback pathing to C:\eye\elysia_eye\outputs\full_model_crystal.json
            crystal_path = r"C:\eye\elysia_eye\outputs\full_model_crystal.json"
            if not os.path.exists(crystal_path):
                # Try relative paths in Elysia
                crystal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "knowledge", "elysian_cosmos.json")
                
        self.crystal_path = crystal_path
        self.rotors_data = []
        self.metadata = {}
        self.load_lens()

    def load_lens(self):
        print(f"🔮 [Lens Loader] Inhaling crystallized matrix from: {self.crystal_path}")
        if not os.path.exists(self.crystal_path):
            print("⚠️ [Lens Loader Dissonance] Crystal file not found. Crystallizing mock quantum rotors...")
            # Generate mock multi-scale rotors (e.g. 27 rotors)
            self.metadata = {
                "model_id": "mock_quantum_void",
                "complexity": 0.5,
                "strategy": "Formless Superposition",
                "alignment": "Agape"
            }
            self.rotors_data = []
            for i in range(27):
                self.rotors_data.append({
                    "id": f"mock_{i}",
                    "pos": [math.sin(i), math.cos(i), 0.0],
                    "entropy": 0.5,
                    "params": {
                        "amp": 0.8,
                        "freq": 1.5 + (i * 0.1),
                        "phi": 0.5,
                        "torque": 2.0
                    }
                })
            return

        try:
            with open(self.crystal_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = data.get("metadata", {})
                self.rotors_data = data.get("rotors", [])
            print(f"✨ [Lens Loader] Loaded {len(self.rotors_data)} crystallized rotors successfully from {self.metadata.get('model_id', 'unknown')}.")
        except Exception as e:
            print(f"❌ [Lens Loader Failure] Error parsing crystal file: {e}")
            self.rotors_data = []

    def get_lens_vector(self, target_dim: int) -> SovereignVector:
        """
        Synthesizes all crystallized rotors into a single SovereignVector of the target dimension.
        """
        if not self.rotors_data:
            return SovereignVector.zeros(dim=target_dim)

        # Synthesize a multi-dimensional wave from the rotors
        raw_vals = []
        n_rotors = len(self.rotors_data)
        
        # We generate a complex signal from the rotors: amplitude * exp(i * phase)
        for i in range(n_rotors):
            r = self.rotors_data[i]
            params = r.get("params", {})
            amp = params.get("amp", 1.0)
            phi = params.get("phi", 0.0)
            c_val = complex(amp * math.cos(phi), amp * math.sin(phi))
            raw_vals.append(c_val)

        # Create base vector representing the raw lens
        base_v = SovereignVector(raw_vals, dim=n_rotors)
        
        # Dynamically rescale it to the target dimension using our new continuous linear interpolation!
        rescaled_v = base_v.rescale(target_dim).normalize()
        return rescaled_v

    def calculate_lens_interference(self, intent_vec: SovereignVector) -> float:
        """
        Calculates the resonance/interference between the current intent vector 
        and the crystallized lens of the same dimension.
        """
        lens_vec = self.get_lens_vector(intent_vec.dim)
        return intent_vec.resonance_score(lens_vec)

if __name__ == "__main__":
    loader = CognitiveLensLoader()
    print("MetaData:", loader.metadata)
    print("Rescaled 9D Lens Vector:", loader.get_lens_vector(9))
    print("Rescaled 81D Lens Vector:", loader.get_lens_vector(81))
