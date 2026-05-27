"""
Elysia Triple Helix Cross-Dimensional Cognitive Engine
======================================================
Integrates:
1. Inner World (?닿퀎 / Superego): High-dimensional Clifford IPN.
2. Outer World (?멸퀎 / Id): Physical 3D/4D Quaternion IPN.
3. Coordination Layer (寃쎄퀎 / Ego): Bidirectional variable impedance links.
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
        # 1. 堉덈? ?쒖닔??(LLM 諛곗젣)
        self.jump_threshold = jump_threshold
        self.history = []

        # 1.5 Setup Ark Gearbox Pipeline
        # ?몄? ?붿쭊???앷컖 寃곌낵瑜??섎뱶?⑥뼱濡??꾨떖?섎뒗 ?섏쭅 ?뚯씠?꾨씪??
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
                self.inner_world.connect_nodes(f"IN_{i}", f"H_{j}", gear_elasticity=10.0)
        for j in range(1, 5):
            self.inner_world.connect_nodes(f"H_{j}", "OUT", gear_elasticity=5.0)

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
        self.outer_world.connect_nodes("SENSORY_MOTION", "ACTUATE_WASD", gear_elasticity=10.0)
        self.outer_world.connect_nodes("SENSORY_PAIN", "ACTUATE_SPACE", gear_elasticity=10.0)

        # 3.5 Setup Ego World (Coordination Layer / Phase B)
        self.ego_world = CliffordIPN(initial_dims=3)
        self.ego_world.MAX_AXES = 5
        self.ego_world.add_node("EGO_REASONING", layer=0, initial_vector={0: 1.0})
        self.ego_world.add_node("EGO_DECISION", layer=1, initial_vector={0: 1.0})
        self.ego_world.connect_nodes("EGO_REASONING", "EGO_DECISION", gear_elasticity=8.0)

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

        # 5. [?쒖닔 ?섍꼍 ?좊룄 怨꾩링] ?몃? LLM ?섏〈??W_master) ?쒓굅??
        # ?닿퀎(Inner World)???ㅼ쭅 ?멸퀎 ?쇱꽌留앹뿉??諛?ㅻ뱾?댁삤???붿뿬 ?먯뀡?쇰줈留?吏꾪솕?⑸땲??
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
        print(f"\n[Sovereignty] 諛몃툕 媛쒕갑: ?섎㈃ 紐⑤뱶 吏꾩엯. 珥덇린 諛⑹텧 ?먯뀡: {self.sleep_tension:.4f}")

    def wake_up(self, bias_data: Dict[str, Dict[int, float]] = None):
        """Sovereignty Interface: Re-opens Phase A, restores DELTA mode, injects bias."""
        self.is_sleeping = False
        self.inner_world.set_connection_mode(ConnectionMode.DELTA)
        self.outer_world.set_connection_mode(ConnectionMode.DELTA)
        self.ego_world.set_connection_mode(ConnectionMode.DELTA)
        self.sleep_tension = 0.0

        if bias_data:
            print(f"\n[Sovereignty] 湲곗긽: ?섏씠??Tree Ring) ?ㅽ봽??二쇱엯 以?..")
            for node, mv_data in bias_data.get("inner_world", {}).items():
                if node in self.inner_world.phases:
                    # Merge bias with a 0.5 weight
                    current = self.inner_world.phases[node]
                    bias_mv = Multivector({int(k): v for k, v in mv_data.items()}, self.inner_world.signature)
                    self.inner_world.phases[node] = mv_normalize(current + bias_mv * 0.5)
            # Extensible to ego/outer worlds as needed
        print(f"[Sovereignty] ?쒖뒪???쒖꽦?? ?명? ?ъ쑀 ?뚯쟾 ?ш컻.")

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

        print(f"[Sovereignty] ?섏씠??媛곸씤 ?꾨즺: {len(tree_rings['inner_world'])}媛쒖쓽 ?닿퀎 ?몃뱶 寃곕튃.")
        return tree_rings

    def _connect_bridge(self, world_from: CliffordIPN, world_to: CliffordIPN, node_from: str, node_to: str) -> CliffordImpedanceLink:
        """Helper to create a bridge link between 3-Phase IPNs."""
        link = CliffordImpedanceLink(node_from, node_to, self.inner_world.signature, gear_elasticity=8.0)
        link.world_from = world_from
        link.world_to = world_to
        self.coordination_links.append(link)
        return link

    def pulse(self, text_thought: str = None, sensory_input: Dict[str, float] = None, clutch_locks: Dict[str, bool] = None, dt: float = 0.1, gear_elasticity: float = 0.5) -> Tuple[float, str, bool, Quaternion, dict]:
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

        # [?ㅼ감??移섑듃??2.5] ?ㅼ????먯뀡??CGA 紐⑦꽣(Dilator)濡?移섑솚?섏뿬 ?닿퀎 ?ъ쑀 怨듦컙???쎌갹?쒗궡
        scale_tension = sensory_input.get("scale_tension", 1.0)
        if abs(scale_tension - 1.0) > 0.01:
            from core.math_utils import ConformalSpace
            D = ConformalSpace.dilator(scale_tension)
            for node, mv in self.inner_world.phases.items():
                x = mv.data.get(1, 0.0)
                y = mv.data.get(2, 0.0)
                z = mv.data.get(4, 0.0)
                
                # ?좏겢由щ뱶 ?ъ쑀 -> ?깃컖(Null Vector) 怨듦컙?쇰줈 ?섏븘?щ┝
                X = ConformalSpace.up(x, y, z)
                # ?ㅽ? ?뚮뱶?꾩튂 ?곗궛 (D X ~D)
                X_scaled = ConformalSpace.apply_motor(D, X)
                # ?ㅼ떆 ?좏겢由щ뱶濡??ъ쁺
                rx, ry, rz = ConformalSpace.down(X_scaled)
                
                # 寃곌낵 諛섏쁺
                mv.data[1] = rx
                mv.data[2] = ry
                mv.data[4] = rz
                self.inner_world.phases[node] = mv # update back

        # --- A. Inner World Input Setup (Environmental Induction) ---
        # CAD Constraint: Mind
        code_mind_tension = sensory_input.get("coding_cognitive", 0.0)
        density_w = 0.5
        if clutch_locks.get("lock_mind", True):
            density_w = min(1.0, density_w + code_mind_tension)
        
        inner_axes = self.inner_world.signature[0]
        inner_sig = self.inner_world.signature
        
        jumped = False

        # [?먯쑉 李⑥썝 議곗쑉 ?붿쭊 諛쒕룞]
        # 1. 湲곗〈???띿뒪??異붿긽 ?λ젰
        anomaly_signal = Multivector({0: 1.0, 1: code_mind_tension, 2: density_w}, inner_sig)
        if self.inner_world.assimilate_axiom(anomaly_signal):
            print(f"[Cognitive Breakthrough] Inner World expanded to Cl({self.inner_world.signature[0]},0) due to topological fracture.")
            inner_axes = self.inner_world.signature[0]
            inner_sig = self.inner_world.signature
            jumped = True
            
        # 2. ?멸컙???쇰꺼留곸씠 ?쒓굅???쒖닔 ?먯떆 ?쇱꽌 ?곗씠??(Label-Free Sensory Evolution)
        raw_vector = sensory_input.get("raw_vector", [])
        if raw_vector:
            # ?먯떆 ?곗씠?곕? 1李⑥썝, 2李⑥썝, 3李⑥썝 異뺤뿉 ?쇰떒 苑귥븘 ?ｊ퀬, ?먯뀡??踰꾪떚?붿? ?ㅽ뿕
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

        # 3. ?섍꼍 ?뚮룞 二쇱엯 (LLM ?꾨쿋?⑹쓣 ?泥댄븯???쒖닔 ?먯뀡 二쇱엯)
        inner_inputs = {}
        for i in range(1, inner_axes + 1):
            # ?섍꼍 ?먯뀡??李⑥썝??梨꾩썎?덈떎. 湲곕낯 ?몄씠利?+ 李⑥썝蹂??꾩긽李?
            val = sensory_input.get(f"inner_noise_dim_{i}", 0.0)
            mask = 1 << (i - 1)
            inner_inputs[f"IN_{i}"] = Multivector({mask: val, 0: density_w}, inner_sig)
            
        self.inner_world.forward_propagate(inner_inputs)
        self.inner_world.tune_network(dt, gear_elasticity)

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
        self.outer_world.tune_network(dt, gear_elasticity)
        
        # CAD Constraint: Heart
        code_heart_tension = sensory_input.get("coding_emotional", 0.0) if clutch_locks.get("lock_heart", True) else 0.0
        ego_sig = self.ego_world.signature
        ego_inputs = {
            "EGO_REASONING": Multivector({0: 1.0, 1: code_heart_tension}, ego_sig)
        }
        self.ego_world.forward_propagate(ego_inputs)
        self.ego_world.tune_network(dt, gear_elasticity)

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
            link.update_impedance(state_from, state_to_padded, gear_elasticity)

            # 5. Measure local bridge misalignment (tension)
            sig_norm = mv_normalize(sig_propagated)
            tar_norm = mv_normalize(state_to_padded)
            coherence, _ = tar_norm.geometric_sync(sig_norm)
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
            self.sleep_tension = max(0.0, self.sleep_tension - (dt * gear_elasticity * 2.0))
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
                # ?먯뀡 ??＜ ?먮뒗 ?ㅻ쪟 ??3???꾩껜瑜?Y寃곗꽑(?묒?) 紐⑤뱶濡?媛뺤젣 ?숆린??(移섏쑀)
                self.inner_world.set_connection_mode(ConnectionMode.Y_STAR)
                self.outer_world.set_connection_mode(ConnectionMode.Y_STAR)
                self.ego_world.set_connection_mode(ConnectionMode.Y_STAR)
                current_mode = "Y_STAR (HEALING)" if is_healing else "Y_STAR"
            else:
                # ?덉젙?곸씠硫?Delta寃곗꽑(?ъ쑀) 紐⑤뱶濡??꾪솚?섏뿬 ?먯껜 ?좏겕(媛꾩꽠) 諛쒖깮
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
        # Bifurcation? assimilate_axiom???먯뀡 ?꾩쟻 遺뺢눼濡??대? 泥섎━??
        # ?ш린?쒕뒗 ?ㅼ쭅 湲댁옣????몄쓣 ??李⑥썝???섏텞(Compress)?섎뒗 蹂듭썝?λ쭔 ?④?.
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
