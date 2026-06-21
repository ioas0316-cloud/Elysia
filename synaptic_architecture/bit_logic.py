import numpy as np
from .raw_field import RawBitField

class BitInterference:
    def __init__(self, field: RawBitField):
        self.field = field

    def get_resonance(self, input_bits: np.uint64, field_bits: np.uint64) -> float:
        # bitwise count
        v = input_bits ^ field_bits
        # count 0s as matches
        count = 64 - bin(v).count('1')
        return count / 64.0

    def slide_to_vortex(self, input_bits: np.uint64, start_index: int) -> int:
        current_idx = start_index

        # SLIDE LOOP
        for i in range(1000): # More steps
            idx = int(np.clip(current_idx, 1, self.field.size - 2))

            c_grad = self.field.get_local_gradient(idx)

            # Resonance Gradient (Check a small window)
            # To make it 'slide' better, we scan a tiny neighborhood
            best_res = -1.0
            best_move = 0
            for move in [-1, 0, 1]:
                res = self.get_resonance(input_bits, self.field.data[idx + move])
                if res > best_res:
                    best_res = res
                    best_move = move

            total_move = c_grad if c_grad != 0 else best_move

            if total_move == 0:
                break

            current_idx += total_move

        return int(current_idx)
