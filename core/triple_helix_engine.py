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
from core.clifford_impedance_network import CliffordIPN, CliffordImpedanceLink, mv_normalize, mv_norm, ConnectionMode
from core.enneagram_rotor import EnneagramRotor

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

        # 3. Setup Outer World (Somatic Layer - locked at Cl(4,0))
        self.outer_world = CliffordIPN(initial_dims=4)
        self.outer_world.MAX_AXES = 4 # Hard lock to 4D to support code tension
        self.outer_world.MIN_AXES = 3
        
        # Somatic Sensory nodes (inputs)
        self.outer_world.add_node("SENSORY_MOTION", layer=0, initial_vector={1: 1.0}) # e1
        self.outer_world.add_node("SENSORY_PAIN", layer=0, initial_vector={2: 1.0})    # e2
        self.outer_world.add_node("SENSORY_VISION", layer=0, initial_vector={4: 1.0})  # e3
        self.outer_world.add_node("SENSORY_CODE", layer=0, initial_vector={8: 1.0})    # e4
        # Somatic Action nodes (outputs)
        self.outer_world.add_node("ACTUATE_WASD", layer=1, initial_vector={0: 1.0})
        self.outer_world.add_node("ACTUATE_SPACE", layer=1, initial_vector={0: 1.0})
        
        # Connect Somatic inputs to Somatic outputs
        self.outer_world.connect_nodes("SENSORY_MOTION", "ACTUATE_WASD", initial_R=10.0)
        self.outer_world.connect_nodes("SENSORY_PAIN", "ACTUATE_SPACE", initial_R=10.0)

        # 3.5 Setup Ego World (Coordination Layer / Phase B)
        self.ego_world = CliffordIPN(initial_dims=3)
        self.ego_world.MAX_AXES = 5
        self.ego_world.add_node("EGO_REASONING", layer=0, initial_vector={0: 1.0})
        self.ego_world.add_node("EGO_DECISION", layer=1, initial_vector={0: 1.0})
        self.ego_world.connect_nodes("EGO_REASONING", "EGO_DECISION", initial_R=8.0)

        # 4. Setup Coordination Layer (Bridging Links)
        # These links operate under the signature of the Inner World (since it is higher dimensional)
        self.coordination_links: List[CliffordImpedanceLink] = []
        
        # Bridge 1: Phase C (Inner) -> Phase B (Ego)
        self.link_out_ego = self._connect_bridge(self.inner_world, self.ego_world, "OUT", "EGO_REASONING")
        
        # Bridge 2: Phase B (Ego) -> Phase A (Outer)
        self.link_ego_wasd = self._connect_bridge(self.ego_world, self.outer_world, "EGO_DECISION", "ACTUATE_WASD")
        self.link_ego_space = self._connect_bridge(self.ego_world, self.outer_world, "EGO_DECISION", "ACTUATE_SPACE")
        
        # Bridge 3: Phase A (Outer) -> Phase C (Inner)
        self.link_pain_h1 = self._connect_bridge(self.outer_world, self.inner_world, "SENSORY_PAIN", "H_1")
        self.link_motion_h2 = self._connect_bridge(self.outer_world, self.inner_world, "SENSORY_MOTION", "H_2")

        # 5. Deterministic Master Projection W_master in R^(384 x 8)
        np.random.seed(42)
        self.W_master = np.random.randn(384, 8) / np.sqrt(384)

    def _connect_bridge(self, world_from: CliffordIPN, world_to: CliffordIPN, node_from: str, node_to: str) -> CliffordImpedanceLink:
        """Helper to create a bridge link between 3-Phase IPNs."""
        link = CliffordImpedanceLink(node_from, node_to, self.inner_world.signature, initial_R=8.0)
        link.world_from = world_from
        link.world_to = world_to
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

    def pulse(self, text_thought: str, sensory_input: Dict[str, float], clutch_locks: Dict[str, bool] = None, dt: float = 0.1, lr: float = 0.5) -> Tuple[float, str, bool, Quaternion, dict]:
        """
        Executes a Triple Helix loop cycle:
        1. Inner World updates based on text thought.
        2. Outer World updates based on somatic sensory input (motion, pain).
        3. Information propagates across Coordination Bridge links.
        4. Links adapt. Average coordination tension is evaluated.
        5. Bifurcation/compression triggered in Inner World and bridge links.
        6. Actuation state projected to unit Quaternion.
        """
        if clutch_locks is None:
            clutch_locks = {"lock_body": True, "lock_mind": True, "lock_heart": True}

        # --- A. Inner World Input Setup ---
        emb = self.encoder.encode([text_thought])[0]
        density_w = self.get_text_density(text_thought)
        
        # CAD Constraint: Mind
        code_mind_tension = sensory_input.get("coding_cognitive", 0.0)
        if clutch_locks.get("lock_mind", True):
            density_w = min(1.0, density_w + code_mind_tension)
        
        inner_axes = self.inner_world.signature[0]
        inner_sig = self.inner_world.signature
        
        # [자율 차원 조율 엔진 발동]
        # 인지적 장력(코드의 거대한 구조적 변화나 난해한 텍스트 사유)이 임계치를 넘으면 새로운 공리로 취급
        if code_mind_tension > 0.4:
            # 텍스트와 텐션이 섞인 낯선 파동 생성
            anomaly_signal = Multivector({0: 1.0, 1: code_mind_tension, 2: density_w}, inner_sig)
            if self.inner_world.assimilate_axiom(anomaly_signal, threshold=0.15):
                print(f"[Cognitive Breakthrough] Inner World expanded to Cl({self.inner_world.signature[0]},0) due to unexplainable axiom tension.")
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

        # --- B. Outer World & Ego World Input Setup ---
        motion = sensory_input.get("motion_entropy", 0.0)
        pain = sensory_input.get("pain_level", 0.0)
        vision = sensory_input.get("visual_entropy", 0.0)
        
        # CAD Constraint: Body
        code_body_tension = sensory_input.get("coding_somatic", 0.0) if clutch_locks.get("lock_body", True) else 0.0
        
        outer_sig = self.outer_world.signature
        outer_inputs = {
            "SENSORY_MOTION": Multivector({1: motion}, outer_sig), # e1
            "SENSORY_PAIN": Multivector({2: pain}, outer_sig),     # e2
            "SENSORY_VISION": Multivector({4: vision}, outer_sig), # e3
            "SENSORY_CODE": Multivector({8: code_body_tension}, outer_sig) # e4
        }
        self.outer_world.forward_propagate(outer_inputs)
        self.outer_world.tune_network(dt, lr)
        
        # CAD Constraint: Heart
        code_heart_tension = sensory_input.get("coding_emotional", 0.0) if clutch_locks.get("lock_heart", True) else 0.0
        ego_sig = self.ego_world.signature
        ego_inputs = {
            "EGO_REASONING": Multivector({0: 1.0, 1: code_heart_tension}, ego_sig)
        }
        self.ego_world.forward_propagate(ego_inputs)
        self.ego_world.tune_network(dt, lr)

        # --- C. Cross-Dimensional Coordination Layer Propagation ---
        coord_tension = 0.0
        
        for link in self.coordination_links:
            wf = link.world_from
            wt = link.world_to
            
            # 1. Resolve source multivector (padded to inner_sig)
            state_from = Multivector(wf.phases[link.node_from].data, inner_sig)

            # 2. Propagate through bridge link
            link.I = state_from
            sig_propagated = link.propagate(state_from)

            # 3. Inject and accumulate into destination node
            if wt == self.outer_world:
                discard_mask = ~7 # Truncate to Cl(3,0)
                truncated_data = {k: v for k, v in sig_propagated.data.items() if (k & discard_mask) == 0}
                sig_dest = Multivector(truncated_data, outer_sig)
                wt.phases[link.node_to] = mv_normalize(wt.phases[link.node_to] + sig_dest * 0.2)
            else:
                wt.phases[link.node_to] = mv_normalize(wt.phases[link.node_to] + sig_propagated * 0.2)
                
            state_to_padded = Multivector(wt.phases[link.node_to].data, inner_sig)

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

        # --- Y/Delta Dynamic Scheduler & Trinity Healing ---
        # Check for NaN or exploded tension in any phase
        is_healing = False
        tensions = [self.inner_world.tension, self.outer_world.tension, self.ego_world.tension]
        if math.isnan(sum(tensions)) or max(tensions) > self.jump_threshold * 2.0:
            is_healing = True

        if is_healing or avg_tension > self.jump_threshold * 0.8:
            # 텐션 폭주 또는 오류 시 3상 전체를 Y결선(접지) 모드로 강제 동기화 (치유)
            self.inner_world.set_connection_mode(ConnectionMode.Y_STAR)
            self.outer_world.set_connection_mode(ConnectionMode.Y_STAR)
            self.ego_world.set_connection_mode(ConnectionMode.Y_STAR)
            current_mode = "Y_STAR (HEALING)" if is_healing else "Y_STAR"
        else:
            # 안정적이면 Delta결선(사유) 모드로 전환하여 자체 토크(간섭) 발생
            self.inner_world.set_connection_mode(ConnectionMode.DELTA)
            self.outer_world.set_connection_mode(ConnectionMode.DELTA)
            self.ego_world.set_connection_mode(ConnectionMode.DELTA)
            current_mode = "DELTA"

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

        # Extract Ego Enneagram Phase
        ego_mv = self.ego_world.phases["EGO_DECISION"]
        eqw = ego_mv.data.get(0, 0.0)
        eqx = -ego_mv.data.get(6, 0.0)
        eqy = ego_mv.data.get(5, 0.0)
        eqz = -ego_mv.data.get(3, 0.0)
        enneagram_state = EnneagramRotor.quaternion_to_enneagram(eqw, eqx, eqy, eqz)

        # Store logs
        self.history.append({
            'thought': text_thought,
            'sensory': sensory_input.copy(),
            'inner_axes': self.inner_world.signature[0],
            'tension': avg_tension,
            'mode': current_mode,
            'jumped': jumped,
            'quat': quat,
            'enneagram': enneagram_state
        })

        return avg_tension, current_mode, jumped, quat, enneagram_state
