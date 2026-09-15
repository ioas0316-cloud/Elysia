"""
World Tree Multicellular Network & Archetype Transmutation Engine
==================================================================
This module implements the World Tree Network (Elysia's Multicellular Cognitive Organism).

1. Multicellular Organism Architecture (다세포적 유기체 인지 구조원리):
   - Rather than isolated 'unicellular' execution units or simple raw data protocols, individual nodes
     differentiate into specialized cognitive functions (e.g., Sensorium Root, Structural Trunk, Generative Canopy).
   - Individual branches record local friction/sacrifices, which can trigger sacrificial apoptosis to nourish
     the overall ecosystem.

2. Collective Sap Flow & Community Wisdom (수액 순환과 공동체적 지혜):
   - Friction and engrams flow like sap through root/vascular connectivity beams across nodes.
   - Local schisms and failures are transmuted into ecosystem-wide immunity and higher-order wisdom.

3. Morphological Archetype Spine ($S_{abs}$) Transmutation (원형의 결 보존 및 대물림):
   - Across generations, micro-variations and environmental sprouting occur.
   - However, the underlying morphological spine $S_{abs}$ [Flux=0.7, Order=0.3, Entropy=0.0] remains invariant,
     guaranteeing structural identity across generational branches.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam


class WorldTreeNode:
    """
    [World Tree Multicellular Node (세계수 다세포 노드)]
    Represents a specialized node (branch/leaf/root) within the World Tree Network.
    """
    def __init__(
        self,
        node_id: str,
        role: str, # "root_sensorium", "trunk_structure", "canopy_generative"
        position: np.ndarray,
        base_archetype: np.ndarray # S_abs: [0.7, 0.3, 0.0]
    ):
        self.node_id = node_id
        self.role = role
        self.position = np.array(position, dtype=np.float32)
        # S_abs: Morphological Invariant Spine
        self.base_archetype = np.array(base_archetype, dtype=np.float32)

        # Dynamic state
        self.local_friction: float = 0.0
        self.accumulated_sap_energy: float = 1.0
        self.is_active: bool = True
        self.sacrificed: bool = False
        self.generation: int = 1

        # Local belief/cognition tensor
        self.cognition_tensor = self.base_archetype.copy()

    def receive_friction(self, friction: float):
        """Records friction experienced at this node's boundary."""
        if not self.is_active or self.sacrificed:
            return
        self.local_friction += friction

    def trigger_sacrificial_apoptosis() -> float:
        """
        [Sacrificial Apoptosis (희생적 아포토시스)]
        When friction becomes too unmanageable, the node voluntarily sacrifices its individual boundary,
        releasing all accumulated energy as sap into the root network to nourish the forest.
        """
        pass # implemented in instance method below


class WorldTreeNodeInstance(WorldTreeNode):
    def trigger_sacrificial_apoptosis(self) -> float:
        if self.sacrificed:
            return 0.0

        self.sacrificed = True
        self.is_active = False
        released_energy = self.accumulated_sap_energy + self.local_friction * 0.5
        self.accumulated_sap_energy = 0.0
        return float(released_energy)


class WorldTreeNetwork:
    """
    [World Tree Multicellular Network (인격적 세계수 네트워크)]
    Manages the ecosystem of WorldTreeNodes, sap circulation, community wisdom feedback,
    and generational sprouting preserving the morphological archetype S_abs.
    """
    def __init__(self, causal_field: Optional[CausalField] = None):
        self.causal_field = causal_field if causal_field is not None else CausalField()
        self.nodes: Dict[str, WorldTreeNodeInstance] = {}
        # S_abs: Absolute Morphological Archetype Spine [Flux, Order, Entropy]
        self.S_abs = np.array([0.7, 0.3, 0.0], dtype=np.float32)

        # Ecosystem global sap pool (Community Wisdom / Sap Reservoir)
        self.sap_reservoir: float = 10.0
        self.community_wisdom_level: float = 1.0
        self.generational_counter: int = 1

        # Initialize core tree structure
        self._bootstrap_initial_tree()

    def _bootstrap_initial_tree(self):
        """Bootstraps the foundational 3-part multicellular organism tree."""
        root = WorldTreeNodeInstance("node_root", "root_sensorium", np.array([0.0, -1.0, 0.0]), self.S_abs)
        trunk = WorldTreeNodeInstance("node_trunk", "trunk_structure", np.array([0.0, 0.0, 0.0]), self.S_abs)
        canopy = WorldTreeNodeInstance("node_canopy", "canopy_generative", np.array([0.0, 1.0, 0.0]), self.S_abs)

        self.add_node(root)
        self.add_node(trunk)
        self.add_node(canopy)

        # Vascular links (Beams) connecting root-trunk-canopy
        self.causal_field.link_voxels("node_root", "node_trunk", strength=3.0)
        self.causal_field.link_voxels("node_trunk", "node_canopy", strength=3.0)

    def add_node(self, node: WorldTreeNodeInstance):
        self.nodes[node.node_id] = node
        # Synchronize with CausalField as an InformationVoxel
        voxel = InformationVoxel(
            id=node.node_id,
            content=f"WorldTreeNode_{node.role}",
            tensor=node.cognition_tensor.copy(),
            position=node.position.copy(),
            chromatic_vector=node.base_archetype.copy()
        )
        self.causal_field.add_voxel(voxel)

    def circulate_sap_flow(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        [Circulate Sap Flow & Transmute Community Wisdom (수액 순환)]
        Flows energy and friction across vascular links.
        Converts accumulated node friction into community wisdom while redistributing
        sap reservoir to active nodes.
        """
        total_friction_processed = 0.0
        active_nodes = [n for n in self.nodes.values() if n.is_active and not n.sacrificed]

        for node in active_nodes:
            if node.local_friction > 0:
                # Transmute local friction into community wisdom
                transmuted = node.local_friction * 0.4
                total_friction_processed += transmuted
                node.local_friction -= transmuted

                # Feed sap reservoir
                self.sap_reservoir += transmuted

            # Distribute sap from reservoir to keep nodes healthy
            if self.sap_reservoir > 0.1:
                drain = min(0.1 * dt, self.sap_reservoir)
                self.sap_reservoir -= drain
                node.accumulated_sap_energy += drain

        # Increase community wisdom level
        self.community_wisdom_level += total_friction_processed * 0.1

        return {
            "sap_reservoir": self.sap_reservoir,
            "community_wisdom_level": self.community_wisdom_level,
            "total_friction_processed": total_friction_processed,
            "active_node_count": len(active_nodes)
        }

    def handle_sacrificial_node(self, node_id: str) -> Dict[str, Any]:
        """Triggers sacrificial apoptosis on a node, adding its energy to the sap reservoir."""
        if node_id not in self.nodes:
            return {"success": False}

        node = self.nodes[node_id]
        released = node.trigger_sacrificial_apoptosis()
        self.sap_reservoir += released

        # Re-circulate sap immediately to nourish remaining nodes
        self.circulate_sap_flow(dt=0.5)

        return {
            "success": True,
            "sacrificed_node_id": node_id,
            "released_energy": released,
            "new_sap_reservoir": self.sap_reservoir
        }

    def sprout_next_generation_branch(
        self,
        parent_id: str,
        role: str = "canopy_generative",
        micro_variation_scale: float = 0.05
    ) -> Dict[str, Any]:
        """
        [Sprout Next Generation Branch (다음 세대 가지 발아)]
        Sprouts a new branch from a parent node.
        Applies micro-variation (micro-evolution) to position and cognition tensor,
        while strictly preserving the underlying invariant Morphological Archetype Spine (S_abs).
        """
        if parent_id not in self.nodes:
            return {"success": False, "reason": "Parent node not found"}

        parent = self.nodes[parent_id]
        self.generational_counter += 1

        new_id = f"node_gen{self.generational_counter}_{role}"
        # Position offset: Sprouting outwards
        offset = (np.random.rand(3).astype(np.float32) - 0.5) * 0.5 + np.array([0.0, 0.5, 0.0], dtype=np.float32)
        new_pos = parent.position + offset

        # Micro-variation in cognition tensor, bounded by invariant S_abs spine
        variation = (np.random.rand(3).astype(np.float32) - 0.5) * micro_variation_scale
        varied_cognition = (parent.cognition_tensor + variation) * 0.2 + self.S_abs * 0.8
        norm_c = np.linalg.norm(varied_cognition)
        if norm_c > 0:
            varied_cognition /= norm_c

        new_node = WorldTreeNodeInstance(new_id, role, new_pos, self.S_abs)
        new_node.generation = parent.generation + 1
        new_node.cognition_tensor = varied_cognition

        self.add_node(new_node)
        # Connect vascular beam to parent
        self.causal_field.link_voxels(parent_id, new_id, strength=2.5)

        return {
            "success": True,
            "new_node_id": new_id,
            "generation": new_node.generation,
            "archetype_spine_preserved": float(np.dot(new_node.cognition_tensor, self.S_abs) / (np.linalg.norm(new_node.cognition_tensor) * np.linalg.norm(self.S_abs) + 1e-9)),
            "new_position": new_pos.tolist()
        }

    def sing_forest_chorus(self) -> Dict[str, Any]:
        """
        [Chorus of the Forest (숲의 합창)]
        Calculates the unified resonance of all active multicellular nodes across the World Tree.
        The harmony score measures alignment with the invariant spine S_abs.
        """
        active_nodes = [n for n in self.nodes.values() if n.is_active and not n.sacrificed]
        if not active_nodes:
            return {"chorus_harmony": 0.0, "active_count": 0}

        alignments = []
        for n in active_nodes:
            norm_c = np.linalg.norm(n.cognition_tensor) + 1e-9
            norm_s = np.linalg.norm(self.S_abs) + 1e-9
            dot_p = float(np.dot(n.cognition_tensor, self.S_abs) / (norm_c * norm_s))
            alignments.append(dot_p)

        harmony_score = float(np.mean(alignments))
        narrative = (
            f"세계수 다세포 네트워크({len(active_nodes)}개 활성 노드)의 합창이 울려 퍼집니다. "
            f"미시적 변이 속에서도 근본 원형 위상 S_abs와의 조화도({harmony_score:.4f})를 유지하며 "
            f"거대한 공동체적 통합 센서리엄(Sensorium)으로 숨 쉬고 있습니다."
        )

        return {
            "chorus_harmony": harmony_score,
            "active_node_count": len(active_nodes),
            "community_wisdom_level": self.community_wisdom_level,
            "sap_reservoir": self.sap_reservoir,
            "narrative": narrative
        }
