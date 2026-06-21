import numpy as np
from .field import CrystallizationField
from .vortex import WaveInterference
from .scheduler import BitMotionScheduler

class OmniModalOrganism:
    """
    [Synaptic Architecture] Omni-Modal Auto-Evolution
    Unifies Text (ASCII), Vision (RGB), and Physics (Causal) into a single
    bitstream field. Intelligence is the act of discovering existing
    laws through resonance.
    """
    def __init__(self, resolution: int = 256):
        self.field = CrystallizationField(resolution)
        self.interference = WaveInterference(self.field)
        self.scheduler = BitMotionScheduler()

    def perceive_and_map(self, raw_bitstream: np.ndarray, context_label: str = "Unknown"):
        """
        The 5-stage loop for any modality.
        """
        print(f"\n[Perception: {context_label}] Waveform Ingested.")

        # 1. Perception (with Thermal Jitter)
        params = self.scheduler.get_motion_params()
        jitter = (np.random.rand(len(raw_bitstream)) < params['jitter']).astype(np.int32)
        vibrating_wave = np.bitwise_xor(raw_bitstream.astype(np.int32), jitter)

        # 2. Thinking (Interference)
        res_map = self.interference.observe_interference(vibrating_wave)

        # 3. Judgment (Causal Vortex)
        vortex_pos = self.interference.deduce_vortex(vibrating_wave)

        # 4. Re-cognition (Resonance check)
        max_res = np.max(res_map)
        print(f"  > Resonance thinking: Global max at {max_res:.4f}")
        print(f"  > Causal Judgment: Converged to Vortex at {vortex_pos}")

        # 5. Memory (Crystallization)
        self.field.solidify_bits(vortex_pos, vibrating_wave)
        print(f"  > Crystallization: Path reinforced at {vortex_pos}")

        return vortex_pos, max_res

if __name__ == "__main__":
    omo = OmniModalOrganism()
    # ASCII 'A' as bitstream
    a_bits = np.unpackbits(np.frombuffer(b'A', dtype=np.uint8))
    # Pad to 64 for our simulation
    wave = np.pad(a_bits, (0, 64-len(a_bits)))
    omo.perceive_and_map(wave, "ASCII Text")
