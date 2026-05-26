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
from core.math_utils import Quaternion, Multivector
from core.clifford_impedance_network import CliffordIPN, CliffordImpedanceLink, mv_normalize, mv_norm, ConnectionMode
from core.enneagram_rotor import EnneagramRotor
from core.atlantis_clifford_bridge import AtlantisCliffordSystem

class TripleHelixEngine:
    def __init__(self, jump_threshold=0.5):
        # 1. 뼈대 순수화 (LLM 배제)
        self.jump_threshold = jump_threshold
        self.history = []

        # 1.5 Setup Ark Gearbox Pipeline
        # 인지 엔진의 생각 결과를 하드웨어로 전달하는 수직 파이프라인
        self.ark_gearbox = AtlantisCliffordSystem()

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

        # 5. [순수 환경 유도 계층] 외부 LLM 의존성(W_master) 제거됨.
        # 내계(Inner World)는 오직 외계 센서망에서 밀려들어오는 잔여 텐션으로만 진화합니다.
        # 6. Sovereignty Interface State
        self.is_sleeping = False
        self.sleep_tension = 0.0

    def decide_sleep(self):
        """Sovereignty Interface: Closes Phase A, forces Y-STAR mode, begins tension bleeding."""
        self.is_sleeping = True
        self.inner_world.set_connection_mode(ConnectionMode.Y_STAR)
        self.outer_world.set_connection_mode(ConnectionMode.Y_STAR)
        self.ego_world.set_connection_mode(ConnectionMode.Y_STAR)
        # Calculate current aggregate tension to bleed
        self.sleep_tension = (self.inner_world.tension + self.outer_world.tension + self.ego_world.tension) / 3.0
        print(f"\n[Sovereignty] 밸브 개방: 수면 모드 진입. 초기 방출 텐션: {self.sleep_tension:.4f}")

    def wake_up(self, bias_data: Dict[str, Dict[int, float]] = None):
        """Sovereignty Interface: Re-opens Phase A, restores DELTA mode, injects bias."""
        self.is_sleeping = False
        self.inner_world.set_connection_mode(ConnectionMode.DELTA)
        self.outer_world.set_connection_mode(ConnectionMode.DELTA)
        self.ego_world.set_connection_mode(ConnectionMode.DELTA)
        self.sleep_tension = 0.0

        if bias_data:
            print(f"\n[Sovereignty] 기상: 나이테(Tree Ring) 오프셋 주입 중...")
            for node, mv_data in bias_data.get("inner_world", {}).items():
                if node in self.inner_world.phases:
                    # Merge bias with a 0.5 weight
                    current = self.inner_world.phases[node]
                    bias_mv = Multivector({int(k): v for k, v in mv_data.items()}, self.inner_world.signature)
                    self.inner_world.phases[node] = mv_normalize(current + bias_mv * 0.5)
            # Extensible to ego/outer worlds as needed
        print(f"[Sovereignty] 시스템 활성화: 델타 사유 회전 재개.")

    def freeze_geodesic(self) -> Dict[str, Dict[int, float]]:
        """Sovereignty Interface: Extracts the strongest resonance multivectors (Tree Rings)."""
        tree_rings = {"inner_world": {}, "ego_world": {}}

        # Extract prominent nodes from inner world
        for node, mv in self.inner_world.phases.items():
            if mv_norm(mv) > 0.1: # Only save significant phases
                tree_rings["inner_world"][node] = mv.data.copy()

        # Extract prominent nodes from ego world
        for node, mv in self.ego_world.phases.items():
            if mv_norm(mv) > 0.1:
                tree_rings["ego_world"][node] = mv.data.copy()

        print(f"[Sovereignty] 나이테 각인 완료: {len(tree_rings['inner_world'])}개의 내계 노드 결빙.")
        return tree_rings

    def _connect_bridge(self, world_from: CliffordIPN, world_to: CliffordIPN, node_from: str, node_to: str) -> CliffordImpedanceLink:
        """Helper to create a bridge link between 3-Phase IPNs."""
        link = CliffordImpedanceLink(node_from, node_to, self.inner_world.signature, initial_R=8.0)
        link.world_from = world_from
        link.world_to = world_to
        self.coordination_links.append(link)
        return link

    def pulse(self, text_thought: str = None, sensory_input: Dict[str, float] = None, clutch_locks: Dict[str, bool] = None, dt: float = 0.1, lr: float = 0.5) -> Tuple[float, str, bool, Quaternion, dict]:
        """
        Executes a Triple Helix loop cycle (Purified Autopoiesis):
        1. Inner World updates based purely on abstract semantic tension (bypassing LLM).
        2. Outer World updates based on somatic sensory input.
        3. Information propagates across Coordination Bridge links.
        4. Links adapt. Average coordination tension is evaluated.
        5. Bifurcation/compression triggered in Inner World.
        6. Actuation state projected to unit Quaternion.
        """
        if clutch_locks is None:
            clutch_locks = {"lock_body": True, "lock_mind": True, "lock_heart": True}

        if sensory_input is None:
            sensory_input = {}

        # --- A. Inner World Input Setup (Environmental Induction) ---
        # CAD Constraint: Mind
        code_mind_tension = sensory_input.get("coding_cognitive", 0.0)
        density_w = 0.5
        if clutch_locks.get("lock_mind", True):
            density_w = min(1.0, density_w + code_mind_tension)
        
        inner_axes = self.inner_world.signature[0]
        inner_sig = self.inner_world.signature
        
        jumped = False

        # [자율 차원 조율 엔진 발동]
        # 1. 기존의 텍스트 추상 장력
        anomaly_signal = Multivector({0: 1.0, 1: code_mind_tension, 2: density_w}, inner_sig)
        if self.inner_world.assimilate_axiom(anomaly_signal):
            print(f"[Cognitive Breakthrough] Inner World expanded to Cl({self.inner_world.signature[0]},0) due to topological fracture.")
            inner_axes = self.inner_world.signature[0]
            inner_sig = self.inner_world.signature
            jumped = True
            
        # 2. 인간의 라벨링이 제거된 순수 원시 센서 데이터 (Label-Free Sensory Evolution)
        raw_vector = sensory_input.get("raw_vector", [])
        if raw_vector:
            # 원시 데이터를 1차원, 2차원, 3차원 축에 일단 꽂아 넣고, 텐션을 버티는지 실험
            raw_data = {0: 1.0}
            for i, val in enumerate(raw_vector):
                mask = 1 << i
                raw_data[mask] = val
            
            raw_signal = Multivector(raw_data, inner_sig)
            if self.inner_world.assimilate_axiom(raw_signal):
                print(f"[Sensory Evolution] Inner World expanded to Cl({self.inner_world.signature[0]},0) to accommodate raw unlabelled vector.")
                inner_axes = self.inner_world.signature[0]
                inner_sig = self.inner_world.signature
                jumped = True

        # 3. 환경 파동 주입 (LLM 임베딩을 대체하는 순수 텐션 주입)
        inner_inputs = {}
        for i in range(1, inner_axes + 1):
            # 환경 텐션이 차원을 채웁니다. 기본 노이즈 + 차원별 위상차
            val = sensory_input.get(f"inner_noise_dim_{i}", 0.0)
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
        outer_sig = self.outer_world.signature
        if self.is_sleeping:
            # Block external inputs during sleep, force to 0
            outer_inputs = {
                "SENSORY_MOTION": Multivector({1: 0.0}, outer_sig),
                "SENSORY_PAIN": Multivector({2: 0.0}, outer_sig),
                "SENSORY_VISION": Multivector({4: 0.0}, outer_sig)
            }
        else:
            motion = sensory_input.get("motion_entropy", 0.0)
            pain = sensory_input.get("pain_level", 0.0)
            vision = sensory_input.get("visual_entropy", 0.0)

            outer_inputs = {
                "SENSORY_MOTION": Multivector({1: motion}, outer_sig), # e1
                "SENSORY_PAIN": Multivector({2: pain}, outer_sig),     # e2
                "SENSORY_VISION": Multivector({4: vision}, outer_sig)  # e3
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
        is_healing = False
        current_mode = "DELTA"

        if self.is_sleeping:
            current_mode = "Y_STAR (SLEEPING)"
            # Bleed tension to 0
            self.sleep_tension = max(0.0, self.sleep_tension - (dt * lr * 2.0))
            # Smoothly decay inner/outer/ego network phase magnitudes
            for world in [self.inner_world, self.outer_world, self.ego_world]:
                for node in world.phases:
                    world.phases[node] = world.phases[node] * 0.95
            avg_tension = self.sleep_tension
        else:
            # Check for NaN or exploded tension in any phase
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

        # --- C.5 Ark Gearbox Pipeline Injection ---
        # Convert tension directly to an angle (e.g. scale tension 0.0~2.0 -> 0~180 degrees intent)
        intent_angle = min(avg_tension / self.jump_threshold * 90.0, 180.0)

        # Determine mode string for the gearbox
        if current_mode == ConnectionMode.DELTA:
            gearbox_mode = "DELTA"
        else:
            gearbox_mode = "WYE"

        # Apply intent to Layer 10
        self.ark_gearbox.apply_agent_intent(intent_angle, gearbox_mode)

        # You can also fetch the dashboard to log or react further
        dash = self.ark_gearbox.get_dashboard_needle()

        # --- D. Dynamic Mitosis / Bifurcation ---
        # Bifurcation은 assimilate_axiom의 텐션 누적 붕괴로 이미 처리됨.
        # 여기서는 오직 긴장이 풀렸을 때 차원을 수축(Compress)하는 복원력만 남김.
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
            'sensory': sensory_input.copy(),
            'inner_axes': self.inner_world.signature[0],
            'tension': avg_tension,
            'mode': current_mode,
            'jumped': jumped,
            'quat': quat,
            'enneagram': enneagram_state
        })

        return avg_tension, current_mode, jumped, quat, enneagram_state
