"""
Elysia Thought Phase Alignment Engine
======================================
Projects user's thought texts into 4D Quaternion vectors using SentenceTransformers
and maps rotation diffs to dimensional shifts.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

class ThoughtAlignerMath:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load lightweight sentence transformer model
        self.encoder = SentenceTransformer(model_name)
        # Use a fixed projection matrix to reduce 384D embedding to 3D coordinates
        np.random.seed(42)
        self.projection_matrix = np.random.randn(384, 3) / np.sqrt(384)

    def get_text_density(self, text: str) -> float:
        """Determines the 'density' or scalar aspect (w) of the text."""
        words = text.split()
        if not words:
            return 0.5
        unique_words = set(words)
        density = len(unique_words) / len(words)
        length_factor = min(len(words) / 50.0, 1.0)
        return (density + length_factor) / 2.0

    def text_to_quaternion(self, text: str) -> np.ndarray:
        """Maps a text input to a 4D Quaternion array [w, x, y, z]."""
        w = self.get_text_density(text)
        emb = self.encoder.encode([text])[0]
        
        # Projection to 3D
        xyz = np.dot(emb, self.projection_matrix)
        q = np.array([w, xyz[0], xyz[1], xyz[2]])
        
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        return q

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
        dot_product = np.abs(np.dot(q1, q2))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return float(2.0 * np.arccos(dot_product))

    @staticmethod
    def get_direction_vector(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        q1_conj = ThoughtAlignerMath.quaternion_conjugate(q1)
        q_diff = ThoughtAlignerMath.quaternion_multiply(q2, q1_conj)
        v = q_diff[1:4]
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

class ThoughtAlignerEngine:
    def __init__(self, jump_threshold=0.8):
        self.math = ThoughtAlignerMath()
        self.jump_threshold = jump_threshold
        self.current_q = None
        self.fractal_depth = 1
        self.history = []

    def process_thought(self, text: str) -> tuple[float, bool, np.ndarray]:
        """
        Process a new thought, calculate phase shift, and handle potential fractal jumps.
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
            jumped = True
            direction_v = self.math.get_direction_vector(self.current_q, new_q)
            self.fractal_depth += 1

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
