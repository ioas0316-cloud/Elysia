"""
Elysia Triple Helix Cross-Dimensional Cognitive Engine
======================================================
Integrates:
1. Inner World (내계 / Superego): High-dimensional Clifford IPN.
2. Outer World (외계 / Id): Physical 3D/4D Quaternion IPN.
3. Coordination Layer (경계 / Ego): Bidirectional variable impedance links.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['HF_SKIP_TF_IMPORT'] = '1'

import math
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from core.math_utils import Quaternion, Multivector
from core.clifford_impedance_network import CliffordIPN, CliffordImpedanceLink, mv_normalize, mv_norm

class TripleHelixEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', jump_threshold=0.5):
        # 1. Load Sentence Transformer
        self.encoder = SentenceTransformer(model_name)
        self.jump_threshold = jump_threshold
        self.history = []

        # 2. Setup Inner World (Cognitive Layer - starts at Cl(3,0), mutable)
        self.inner_world = CliffordIPN(initial_dims=3)
        # Inputs: IN_1 to IN_8
        for i in range(1, 9):
            self.inner_world.add_node(f"IN_{i}", layer=0, initial_vector={1 << (i - 1): 1.0})
        # Hidden: H_1 to H_4
        for j in range(1, 5):
            self.inner_world.add_node(f"H_{j}", layer=1, initial_vector={0: 1.0})
        # Output: OUT
        self.inner_world.add_node("OUT", layer=2, initial_vector={0: 1.0})
        
        # Connect Inner World
        for i in range(1, 9):
            for j in range(1, 5):
                self.inner_world.connect_nodes(f"IN_{i}", f"H_{j}", initial_R=10.0)
        for j in range(1, 5):
            self.inner_world.connect_nodes(f"H_{j}", "OUT", initial_R=5.0)

        # 3. Setup Outer World (Somatic Layer - locked at Cl(3,0))
        self.outer_world = CliffordIPN(initial_dims=3)
        self.outer_world.MAX_AXES = 3 # Hard lock to 3D/Quaternion
        self.outer_world.MIN_AXES = 3
        
        # Somatic Sensory nodes (inputs)
        self.outer_world.add_node("SENSORY_MOTION", layer=0, initial_vector={1: 1.0}) # e1
        self.outer_world.add_node("SENSORY_PAIN", layer=0, initial_vector={2: 1.0})    # e2
        # Somatic Action nodes (outputs)
        self.outer_world.add_node("ACTUATE_WASD", layer=1, initial_vector={0: 1.0})
        self.outer_world.add_node("ACTUATE_SPACE", layer=1, initial_vector={0: 1.0})
        
        # Connect Somatic inputs to Somatic outputs
        self.outer_world.connect_nodes("SENSORY_MOTION", "ACTUATE_WASD", initial_R=10.0)
        self.outer_world.connect_nodes("SENSORY_PAIN", "ACTUATE_SPACE", initial_R=10.0)

        # 4. Setup Coordination Layer (Bridging Links)
        # These links operate under the signature of the Inner World (since it is higher dimensional)
        self.coordination_links: List[CliffordImpedanceLink] = []
        
        # Bridge 1: Inner World Output -> Outer World Actions (Intention-Actuation)
        self.link_out_wasd = self._connect_bridge("OUT", "ACTUATE_WASD", is_inner_to_outer=True)
        self.link_out_space = self._connect_bridge("OUT", "ACTUATE_SPACE", is_inner_to_outer=True)
        
        # Bridge 2: Outer World Sensory -> Inner World Hidden (Sensory-Cognition Feedback)
        self.link_pain_h1 = self._connect_bridge("SENSORY_PAIN", "H_1", is_inner_to_outer=False)
        self.link_motion_h2 = self._connect_bridge("SENSORY_MOTION", "H_2", is_inner_to_outer=False)

        # 5. Deterministic Master Projection W_master in R^(384 x 8)
        np.random.seed(42)
        self.W_master = np.random.randn(384, 8) / np.sqrt(384)

    def _connect_bridge(self, node_from: str, node_to: str, is_inner_to_outer: bool) -> CliffordImpedanceLink:
        """Helper to create a bridge link under inner_world signature."""
        link = CliffordImpedanceLink(node_from, node_to, self.inner_world.signature, initial_R=8.0)
        # Store metadata for routing direction
        link.is_inner_to_outer = is_inner_to_outer
        self.coordination_links.append(link)
        return link

    def get_text_density(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.5
        unique = set(words)
        density = len(unique) / len(words)
        len_factor = min(len(words) / 50.0, 1.0)
        return (density + len_factor) / 2.0

    def pulse(self, text_thought: str, sensory_input: Dict[str, float], dt: float = 0.1, lr: float = 0.5) -> Tuple[float, bool, Quaternion]:
        """
        Executes a Triple Helix loop cycle:
        1. Inner World updates based on text thought.
        2. Outer World updates based on somatic sensory input (motion, pain).
        3. Information propagates across Coordination Bridge links.
        4. Links adapt. Average coordination tension is evaluated.
        5. Bifurcation/compression triggered in Inner World and bridge links.
        6. Actuation state projected to unit Quaternion.
        """
        # --- A. Inner World Input Setup ---
        emb = self.encoder.encode([text_thought])[0]
        density_w = self.get_text_density(text_thought)
        
        inner_axes = self.inner_world.signature[0]
        inner_sig = self.inner_world.signature
        
        proj = np.dot(emb, self.W_master[:, :inner_axes])
        inner_inputs = {}
        for i in range(1, inner_axes + 1):
            val = proj[i - 1]
            mask = 1 << (i - 1)
            inner_inputs[f"IN_{i}"] = Multivector({mask: val, 0: density_w}, inner_sig)
            
        self.inner_world.forward_propagate(inner_inputs)
        self.inner_world.tune_network(dt, lr)

        # --- B. Outer World Input Setup ---
        motion = sensory_input.get("motion_entropy", 0.0)
        pain = sensory_input.get("pain_level", 0.0)
        
        outer_sig = self.outer_world.signature
        outer_inputs = {
            "SENSORY_MOTION": Multivector({1: motion}, outer_sig), # e1
            "SENSORY_PAIN": Multivector({2: pain}, outer_sig)      # e2
        }
        self.outer_world.forward_propagate(outer_inputs)
        self.outer_world.tune_network(dt, lr)

        # --- C. Cross-Dimensional Coordination Layer Propagation ---
        coord_tension = 0.0
        
        for link in self.coordination_links:
            # 1. Resolve source multivector
            if link.is_inner_to_outer:
                # From Inner (Cl(n,0)) to Outer (Cl(3,0))
                state_from = self.inner_world.phases[link.node_from]
            else:
                # From Outer (Cl(3,0)) to Inner (Cl(n,0))
                outer_mv = self.outer_world.phases[link.node_from]
                # Pad to Inner World signature
                state_from = Multivector(outer_mv.data, inner_sig)

            # 2. Propagate through bridge link (sandwich product)
            # Make sure link current has the right dimensions
            link.I = state_from
            sig_propagated = link.propagate(state_from)

            # 3. Inject and accumulate into destination node
            if link.is_inner_to_outer:
                # Dest is in outer world (Cl(3,0)). Truncate signal dimensions.
                discard_mask = ~7 # Keep only masks 0..7 (first 3 axes)
                truncated_data = {k: v for k, v in sig_propagated.data.items() if (k & discard_mask) == 0}
                sig_dest = Multivector(truncated_data, outer_sig)
                
                # Attract destination phase state
                self.outer_world.phases[link.node_to] = mv_normalize(
                    self.outer_world.phases[link.node_to] + sig_dest * 0.2
                )
                state_to_padded = Multivector(self.outer_world.phases[link.node_to].data, inner_sig)
            else:
                # Dest is in inner world (Cl(n,0)).
                self.inner_world.phases[link.node_to] = mv_normalize(
                    self.inner_world.phases[link.node_to] + sig_propagated * 0.2
                )
                state_to_padded = self.inner_world.phases[link.node_to]

            # 4. Tune bridge link impedance based on alignment
            link.update_impedance(state_from, state_to_padded, lr)

            # 5. Measure local bridge misalignment (tension)
            sig_norm = mv_normalize(sig_propagated)
            tar_norm = mv_normalize(state_to_padded)
            coherence = sig_norm.dot(tar_norm).data.get(0, 0.0)
            coherence = min(1.0, max(-1.0, coherence))
            tension_angle = math.acos(coherence)
            coord_tension += tension_angle

        # Average coordination bridge tension
        avg_tension = coord_tension / len(self.coordination_links)

        # --- D. Dynamic Mitosis / Bifurcation ---
        jumped = False
        if avg_tension > self.jump_threshold:
            # High tension: Bifurcate Inner World and bridge links
            success = self.inner_world.bifurcate()
            if success:
                jumped = True
                new_sig = self.inner_world.signature
                # Expand bridge links signature
                for link in self.coordination_links:
                    link.update_signature(new_sig)
        else:
            # Low tension: Compress if stable
            if avg_tension < self.jump_threshold * 0.3:
                self.inner_world.stable_ticks += 1
                if self.inner_world.stable_ticks >= 5:
                    self.inner_world.compress()
                    new_sig = self.inner_world.signature
                    for link in self.coordination_links:
                        link.update_signature(new_sig)
            else:
                self.inner_world.stable_ticks = 0

        # --- E. Project Action output to Quaternion ---
        # Look at ACTUATE_WASD output phase in outer_world (Cl(3,0))
        act_mv = self.outer_world.phases["ACTUATE_WASD"]
        
        qw = act_mv.data.get(0, 0.0)
        qx = -act_mv.data.get(6, 0.0)
        qy = act_mv.data.get(5, 0.0)
        qz = -act_mv.data.get(3, 0.0)
        
        norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm > 1e-6:
            quat = Quaternion(qw / norm, qx / norm, qy / norm, qz / norm)
        else:
            quat = Quaternion(1.0, 0.0, 0.0, 0.0)

        # Store logs
        self.history.append({
            'thought': text_thought,
            'sensory': sensory_input.copy(),
            'inner_axes': self.inner_world.signature[0],
            'tension': avg_tension,
            'jumped': jumped,
            'quat': quat
        })

        return avg_tension, jumped, quat
