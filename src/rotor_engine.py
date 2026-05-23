import numpy as np
from rotor_math import FractalRotorMath

class FractalRotorEngine:
    def __init__(self, jump_threshold=0.8):
        self.math = FractalRotorMath()
        self.jump_threshold = jump_threshold

        self.current_q = None
        self.fractal_depth = 1

        # Keep track of history for visualization or analysis
        self.history = []

    def process_thought(self, text):
        """
        Process a new text input, calculate phase shift, and handle potential fractal jumps.
        """
        new_q = self.math.text_to_quaternion(text)

        if self.current_q is None:
            self.current_q = new_q
            self.history.append({
                'text': text,
                'q': new_q,
                'theta': 0.0,
                'jumped': False,
                'depth': self.fractal_depth
            })
            return 0.0, False, None

        theta = self.math.quaternion_angle(self.current_q, new_q)
        jumped = False
        direction_v = None

        if theta > self.jump_threshold:
            # [FRACTAL JUMP DETECTED - PHASE COLLAPSE]
            jumped = True
            direction_v = self.math.get_direction_vector(self.current_q, new_q)

            # Increase fractal depth (zoom in)
            self.fractal_depth += 1

            # We DON'T reset the center quaternion, we continue from the new_q
            # but now at a deeper fractal level.

        self.current_q = new_q

        self.history.append({
            'text': text,
            'q': new_q,
            'theta': theta,
            'jumped': jumped,
            'direction': direction_v,
            'depth': self.fractal_depth
        })

        return theta, jumped, direction_v
