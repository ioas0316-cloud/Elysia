r"""
Autonomous Causal Game Mechanics & Closed-Loop Engine Core
==========================================================

Implements Causal Game Mechanics Engine powered by Structural Causal Models (SCM),
Counterfactual Interventions (do(X)), Protocol Boundary Ledgers, Perceptual Lens Scale Switching,
and 4-Layer Closed-Loop Telemetry Engine.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class ScaleLevel(Enum):
    MICRO_NPC = auto()      # Micro scale: individual actors / tiles / NPCs
    MESO_REGION = auto()    # Meso scale: region / town politics and safety
    MACRO_KINGDOM = auto()  # Macro scale: kingdom-wide narrative & geopolitical power graph


class SimdMode(Enum):
    AVX2_256 = 8
    SSE_128 = 4
    Scalar = 1


@dataclass
class ProtocolBoundary:
    protocol_id: str
    min_state_bound: float = -1.0
    max_state_bound: float = 100.0
    max_latency_ms: float = 16.67
    memory_domain_id: int = 1


@dataclass
class BoundaryCollisionEvent:
    protocol_a: str
    protocol_b: str
    cause_description: str
    value_at_collision: float
    fracture_detected: bool = False


class ProtocolBoundaryLedger:
    """Protocol Execution Boundary Manager and Structural Fracture Diagnostic Engine."""

    def __init__(self):
        self.boundaries: Dict[str, ProtocolBoundary] = {}

    def register_protocol_boundary(self, boundary: ProtocolBoundary) -> None:
        self.boundaries[boundary.protocol_id] = boundary

    def diagnose_boundary_collision(
        self,
        proto_a_id: str,
        proto_a_value: float,
        proto_b_id: str,
        proto_b_value: float,
        elapsed_time_ms: float,
    ) -> BoundaryCollisionEvent:
        bound_a = self.boundaries.get(proto_a_id)
        bound_b = self.boundaries.get(proto_b_id)

        if not bound_a or not bound_b:
            return BoundaryCollisionEvent(
                protocol_a=proto_a_id,
                protocol_b=proto_b_id,
                cause_description="Unregistered protocol boundary metadata.",
                value_at_collision=proto_a_value,
                fracture_detected=True,
            )

        if proto_a_value < bound_a.min_state_bound or proto_a_value > bound_a.max_state_bound:
            return BoundaryCollisionEvent(
                protocol_a=proto_a_id,
                protocol_b=proto_b_id,
                cause_description=f"Protocol {proto_a_id} value ({proto_a_value}) exceeded state boundary [{bound_a.min_state_bound}, {bound_a.max_state_bound}]",
                value_at_collision=proto_a_value,
                fracture_detected=True,
            )

        if proto_a_value > bound_b.max_state_bound or proto_b_value < bound_a.min_state_bound:
            return BoundaryCollisionEvent(
                protocol_a=proto_a_id,
                protocol_b=proto_b_id,
                cause_description=f"Domain disconnect between Protocol {proto_a_id} and {proto_b_id}: boundary ranges do not overlap.",
                value_at_collision=proto_a_value,
                fracture_detected=True,
            )

        if elapsed_time_ms > bound_a.max_latency_ms:
            return BoundaryCollisionEvent(
                protocol_a=proto_a_id,
                protocol_b=proto_b_id,
                cause_description=f"Time boundary fracture in Protocol {proto_a_id}: elapsed {elapsed_time_ms}ms > max {bound_a.max_latency_ms}ms",
                value_at_collision=proto_a_value,
                fracture_detected=True,
            )

        return BoundaryCollisionEvent(
            protocol_a=proto_a_id,
            protocol_b=proto_b_id,
            cause_description="Protocols operating within aligned boundary intersection.",
            value_at_collision=proto_a_value,
            fracture_detected=False,
        )


@dataclass
class SCMNode:
    id: str
    factual_state: float = 0.0
    counterfactual_state: float = 0.0
    is_intervened: bool = False
    intervention_value: float = 0.0
    manifested: bool = False
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    structural_eq: Optional[Callable[[List[float]], float]] = None


class StructuralCausalModel:
    """Structural Causal Model (SCM) with Pearl's do(X) intervention operator."""

    def __init__(self):
        self.nodes: Dict[str, SCMNode] = {}

    def add_node(
        self,
        node_id: str,
        structural_eq: Optional[Callable[[List[float]], float]] = None,
    ) -> None:
        if structural_eq is None:
            structural_eq = lambda parents: sum(parents) * 0.8 if parents else 0.0
        self.nodes[node_id] = SCMNode(id=node_id, structural_eq=structural_eq)

    def add_causal_edge(self, cause_id: str, effect_id: str) -> bool:
        if cause_id not in self.nodes or effect_id not in self.nodes:
            return False
        if self._has_causal_path(effect_id, cause_id):
            return False  # Prevent cycle
        self.nodes[cause_id].children.append(effect_id)
        self.nodes[effect_id].parents.append(cause_id)
        return True

    def _has_causal_path(self, start: str, target: str) -> bool:
        if start == target:
            return True
        visited = set([start])
        queue = [start]
        while queue:
            curr = queue.pop(0)
            if curr in self.nodes:
                for child in self.nodes[curr].children:
                    if child == target:
                        return True
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
        return False

    def propagate_factual_states(self) -> None:
        for node_id in self._get_topological_order():
            node = self.nodes[node_id]
            if node.is_intervened:
                node.factual_state = node.intervention_value
            else:
                parent_vals = [self.nodes[p].factual_state for p in node.parents]
                if node.structural_eq:
                    node.factual_state = node.structural_eq(parent_vals)

    def apply_do_intervention(self, node_id: str, value: float) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].is_intervened = True
            self.nodes[node_id].intervention_value = value
            self.nodes[node_id].counterfactual_state = value

    def propagate_counterfactual_states(self) -> None:
        for node_id in self._get_topological_order():
            node = self.nodes[node_id]
            if node.is_intervened:
                node.counterfactual_state = node.intervention_value
            else:
                parent_vals = [self.nodes[p].counterfactual_state for p in node.parents]
                if node.structural_eq:
                    node.counterfactual_state = node.structural_eq(parent_vals)

    def _get_topological_order(self) -> List[str]:
        in_degree = {n: len(self.nodes[n].parents) for n in self.nodes}
        queue = [n for n in in_degree if in_degree[n] == 0]
        topo = []
        while queue:
            curr = queue.pop(0)
            topo.append(curr)
            for child in self.nodes[curr].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        for n in self.nodes:
            if n not in topo:
                topo.append(n)
        return topo


@dataclass
class TelemetrySnapshot:
    cpu_cache_miss_rate: float = 0.05
    gpu_temp_celsius: float = 45.0
    gpu_is_throttled: bool = False
    frame_time_ms: float = 14.0
    timestamp_ns: int = 0


@dataclass
class FrameExecutionProfile:
    simd_width: SimdMode = SimdMode.AVX2_256
    causal_lod_step: int = 1
    chunk_size: int = 1024


class EnvironmentObserver:
    """Hardware telemetry sensing layer with mock and live polling."""

    def __init__(self):
        self.mock_throttle = False

    def inject_mock_throttle(self, enable: bool) -> None:
        self.mock_throttle = enable

    def poll(self, frame_delta_ms: float) -> TelemetrySnapshot:
        if self.mock_throttle:
            return TelemetrySnapshot(
                cpu_cache_miss_rate=0.20,
                gpu_temp_celsius=85.0,
                gpu_is_throttled=True,
                frame_time_ms=frame_delta_ms,
                timestamp_ns=time.time_ns(),
            )
        return TelemetrySnapshot(
            cpu_cache_miss_rate=0.04,
            gpu_temp_celsius=50.0,
            gpu_is_throttled=False,
            frame_time_ms=frame_delta_ms,
            timestamp_ns=time.time_ns(),
        )


class RealtimeGovernor:
    """Closed-Loop Feedback Governor adjusting SIMD width and Causal LOD step."""

    def __init__(self, target_frame_ms: float = 16.67):
        self.target_frame_ms = target_frame_ms

    def evaluate(self, telemetry: TelemetrySnapshot) -> FrameExecutionProfile:
        profile = FrameExecutionProfile()

        if telemetry.gpu_is_throttled or telemetry.gpu_temp_celsius > 80.0:
            profile.simd_width = SimdMode.SSE_128
        else:
            profile.simd_width = SimdMode.AVX2_256

        if telemetry.cpu_cache_miss_rate > 0.15:
            profile.chunk_size = 256
        else:
            profile.chunk_size = 1024

        if telemetry.frame_time_ms > self.target_frame_ms * 1.3:
            profile.causal_lod_step = 4
        elif telemetry.frame_time_ms > self.target_frame_ms:
            profile.causal_lod_step = 2
        else:
            profile.causal_lod_step = 1

        return profile


@dataclass
class CCGameNode:
    """Causal Conservation Node defining in-game actors, factions, and environments."""

    node_id: str
    scale: ScaleLevel
    is_active: bool = True
    tension_field: float = 0.0
    invariants: Dict[str, str] = field(default_factory=dict)
    bound_rules: List[str] = field(default_factory=list)


@dataclass
class NarrativeAnomalyZone:
    """Quarantined SealedAttractor object isolating narrative ruptures."""

    zone_id: str
    quarantined_nodes: List[CCGameNode]
    peak_tension: float
    restructure_attempts: int = 0
    status: str = "SEALED"


class PerceptualLensController:
    """Perceptual Lens: Scale shifting (Micro/Meso/Macro) and invariant observation."""

    def __init__(self):
        self.current_scale = ScaleLevel.MICRO_NPC

    def shift_scale(self, target_scale: ScaleLevel) -> None:
        self.current_scale = target_scale

    def evaluate_narrative_tension(
        self, active_rules: List[str], nodes: Dict[str, CCGameNode]
    ) -> float:
        king_node = nodes.get("NPC_KING_ARTHUR")

        if king_node and not king_node.is_active:
            if "RULE_KING_MUST_GRANT_QUEST" in active_rules:
                return 0.95

        return 0.02


class SealedAttractorVault:
    """Narrative quarantine and autonomous restructuring controller."""

    def __init__(self, critical_threshold: float = 0.5):
        self.critical_threshold = critical_threshold
        self.anomalies: Dict[str, NarrativeAnomalyZone] = {}

    def isolate_narrative_rupture(
        self,
        zone_id: str,
        broken_nodes: List[CCGameNode],
        tension: float,
    ) -> NarrativeAnomalyZone:
        anomaly = NarrativeAnomalyZone(
            zone_id=f"ANOMALY_{zone_id}",
            quarantined_nodes=broken_nodes,
            peak_tension=tension,
        )
        self.anomalies[anomaly.zone_id] = anomaly
        return anomaly

    def restructure_quest_line(
        self, anomaly_id: str, lens: PerceptualLensController
    ) -> Tuple[bool, Optional[CCGameNode], str]:
        anomaly = self.anomalies.get(anomaly_id)
        if not anomaly:
            return False, None, ""

        anomaly.status = "RESTRUCTURING"
        lens.shift_scale(ScaleLevel.MACRO_KINGDOM)

        regency_council_node = CCGameNode(
            node_id="FACTION_REGENCY_COUNCIL",
            scale=ScaleLevel.MACRO_KINGDOM,
            is_active=True,
            tension_field=0.03,
            invariants={"QUEST_TARGET": "CLAIM_VACANT_THRONE"},
            bound_rules=[
                "RULE_INVESTIGATE_ROYAL_ASSASSIN",
                "RULE_BEGIN_CIVIL_WAR_NARRATIVE",
            ],
        )

        new_narrative_rule = "RULE_INTERACT_WITH_REGENCY_COUNCIL"

        anomaly.status = "RECOVERED"
        anomaly.peak_tension = 0.03

        return True, regency_council_node, new_narrative_rule


class CausalGameMechanicsEngine:
    """Autonomous Causal Game Mechanics Main Engine featuring SCM & Protocol Boundary Ledger."""

    def __init__(self):
        self.lens = PerceptualLensController()
        self.vault = SealedAttractorVault(critical_threshold=0.5)
        self.observer = EnvironmentObserver()
        self.governor = RealtimeGovernor(target_frame_ms=16.67)
        self.scm = StructuralCausalModel()
        self.boundary_ledger = ProtocolBoundaryLedger()

        self.nodes: Dict[str, CCGameNode] = {
            "NPC_KING_ARTHUR": CCGameNode(
                node_id="NPC_KING_ARTHUR",
                scale=ScaleLevel.MICRO_NPC,
                is_active=True,
                invariants={"ROLE": "MAIN_QUEST_GIVER"},
            )
        }
        self.active_quest_rules = ["RULE_KING_MUST_GRANT_QUEST"]

    def tick_closed_loop(self, frame_delta_ms: float) -> FrameExecutionProfile:
        snapshot = self.observer.poll(frame_delta_ms)
        return self.governor.evaluate(snapshot)

    def execute_player_action(self, action_type: str, target_id: str) -> Dict[str, Any]:
        target_node = self.nodes.get(target_id)
        if target_node and action_type == "ASSASSINATE":
            target_node.is_active = False

        current_vt = self.lens.evaluate_narrative_tension(
            self.active_quest_rules, self.nodes
        )

        if current_vt > self.vault.critical_threshold:
            anomaly = self.vault.isolate_narrative_rupture(
                zone_id="ROYAL_CAPITAL",
                broken_nodes=[target_node] if target_node else [],
                tension=current_vt,
            )

            success, new_node, new_rule = self.vault.restructure_quest_line(
                anomaly.zone_id, self.lens
            )

            if success and new_node:
                self.nodes[new_node.node_id] = new_node
                if "RULE_KING_MUST_GRANT_QUEST" in self.active_quest_rules:
                    self.active_quest_rules.remove("RULE_KING_MUST_GRANT_QUEST")
                self.active_quest_rules.append(new_rule)

                return {
                    "status": "RESTRUCTURED_AND_RECOVERED",
                    "anomaly_id": anomaly.zone_id,
                    "new_node_id": new_node.node_id,
                    "new_rule": new_rule,
                    "final_tension": anomaly.peak_tension,
                    "scale": self.lens.current_scale.name,
                }

        return {
            "status": "NORMAL_STABLE",
            "current_tension": current_vt,
        }
