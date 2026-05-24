"""
Elysia Clifford Thought Aligner Engine
======================================
Integrates Clifford-IPN with SentenceTransformers text embedding
to provide high-dimensional semantic routing, dynamic dimension split,
and project the output state to 4D Quaternion geometry.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['HF_SKIP_TF_IMPORT'] = '1'

import math
import numpy as np
from sentence_transformers import SentenceTransformer
from core.math_utils import Quaternion, Multivector
from core.clifford_impedance_network import CliffordIPN, mv_normalize, mv_norm

class CliffordThoughtAlignerEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', jump_threshold=0.6, initial_dims=3):
        # 1. Load Sentence Transformer
        self.encoder = SentenceTransformer(model_name)
        
        # 2. Setup Clifford IPN
        self.net = CliffordIPN(initial_dims=initial_dims)
        self.jump_threshold = jump_threshold
        self.history = []
        
        # 3. Deterministic Master Projection Matrix W_master in R^(384 x 8)
        # Consistent column layout: column 1..d remains invariant when dimension expands
        np.random.seed(42)
        self.W_master = np.random.randn(384, 8) / np.sqrt(384)
        
        # 4. Construct network layout
        # Inputs: IN_1 to IN_8
        for i in range(1, 9):
            self.net.add_node(f"IN_{i}", layer=0, initial_vector={1 << (i - 1): 1.0})
            
        # Hidden: H_1 to H_4
        for j in range(1, 5):
            self.net.add_node(f"H_{j}", layer=1, initial_vector={0: 1.0})
            
        # Output: OUT
        self.net.add_node("OUT", layer=2, initial_vector={0: 1.0})
        
        # Connect nodes (Inputs -> Hidden -> Output)
        for i in range(1, 9):
            for j in range(1, 5):
                # Connect input i to hidden j
                self.net.connect_nodes(f"IN_{i}", f"H_{j}", initial_R=10.0)
                
        for j in range(1, 5):
            # Connect hidden j to OUT
            self.net.connect_nodes(f"H_{j}", "OUT", initial_R=5.0)

    def get_text_density(self, text: str) -> float:
        """Determines scalar w coefficient based on text density."""
        words = text.split()
        if not words:
            return 0.5
        unique_words = set(words)
        density = len(unique_words) / len(words)
        length_factor = min(len(words) / 50.0, 1.0)
        return (density + length_factor) / 2.0

    def process_thought(self, text: str, dt: float = 0.1, lr: float = 0.5) -> tuple[float, bool, Quaternion]:
        """
        Process user thought:
        1. Encodes text, projects to active Clifford dimensions.
        2. Feeds vector signals to input nodes.
        3. Propagates and tunes network.
        4. Monitors tension to trigger dynamic dimension split or compression.
        5. Projects output node state to 4D Quaternion for actuation.
        """
        emb = self.encoder.encode([text])[0]
        density_w = self.get_text_density(text)
        
        # Get active dimensions
        active_axes = self.net.signature[0]
        signature = self.net.signature
        
        # Project embedding using first active_axes columns of W_master
        proj = np.dot(emb, self.W_master[:, :active_axes])
        
        # Prepare inputs: IN_i node receives proj[i-1] * e_i + density_w * 1 (scalar)
        inputs = {}
        for i in range(1, active_axes + 1):
            val = proj[i - 1]
            mask = 1 << (i - 1)
            # Input multivector signal: vector component + scalar density
            inputs[f"IN_{i}"] = Multivector({mask: val, 0: density_w}, signature)
            
        # Run forward propagation
        self.net.forward_propagate(inputs)
        
        # Tune network and compute average tension
        tension = self.net.tune_network(dt, lr)
        
        # Monitor tension and handle bifurcation / compression
        jumped = False
        if tension > self.jump_threshold:
            # High tension: Try to bifurcate (Dimension Split)
            success = self.net.bifurcate()
            if success:
                jumped = True
                active_axes = self.net.signature[0]
        else:
            # Low tension: track stability
            if tension < self.jump_threshold * 0.3:
                self.net.stable_ticks += 1
                if self.net.stable_ticks >= 5:
                    self.net.compress()
                    active_axes = self.net.signature[0]
            else:
                self.net.stable_ticks = 0

        # Project output multivector of OUT node to 4D Quaternion
        out_mv = self.net.phases["OUT"]
        
        # Even subalgebra isomorphic map of Cl(3,0) to Quaternion:
        # 1   <-> w
        # e23 <-> -x * i (i.e. x = -e23)
        # e31 <-> y * j  (i.e. y = e31)
        # e12 <-> -z * k (i.e. z = -e12)
        qw = out_mv.data.get(0, 0.0)
        qx = -out_mv.data.get(6, 0.0)
        qy = out_mv.data.get(5, 0.0)
        qz = -out_mv.data.get(3, 0.0)
        
        # Safe normalize to Quaternion
        norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm > 1e-6:
            quat = Quaternion(qw / norm, qx / norm, qy / norm, qz / norm)
        else:
            quat = Quaternion(1.0, 0.0, 0.0, 0.0)

        # Record history
        self.history.append({
            'text': text,
            'axes': active_axes,
            'tension': tension,
            'jumped': jumped,
            'quat': quat
        })
        
        return tension, jumped, quat
