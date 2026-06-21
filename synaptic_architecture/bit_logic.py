import numpy as np

class BitInterference:
    """
    [Synaptic Architecture] Bitstream XOR Interfacial Logic
    Calculates resonance and interference on the temporal axis.
    """
    @staticmethod
    def interference_score(wave_a: np.uint64, wave_b: np.uint64) -> float:
        """
        v ^ v = 0 -> Zero interference (Perfect Resonance).
        Calculates the percentage of matching bits.
        """
        # XOR to find differences
        deficit = wave_a ^ wave_b
        # Fast bit counting (Python's bit_count is O(1) conceptually for fixed bit-width)
        diff_bits = bin(deficit).count('1')
        match_score = (64 - diff_bits) / 64.0
        return match_score

    @staticmethod
    def vortex_pull(target_wave: np.uint64, candidate_waves: np.ndarray) -> int:
        """
        Identifies the index of the highest resonance in a set of waves.
        Used for converging toward a cognitive vortex in a spatial grid.
        """
        best_idx = 0
        max_res = -1.0

        for i, wave in enumerate(candidate_waves):
            res = BitInterference.interference_score(target_wave, wave)
            if res > max_res:
                max_res = res
                best_idx = i

        return best_idx

if __name__ == "__main__":
    bi = BitInterference()
    w1 = np.uint64(0xAAAAAAAAAAAAAAAA)
    w2 = np.uint64(0xAAAAAAAAAAAAAAAF) # 4 bit difference
    print(f"Resonance Score: {bi.interference_score(w1, w2):.4f}")
