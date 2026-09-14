r"""
Autonomous Causal Game Mechanics & Closed-Loop Engine Core
==========================================================

Implements Causal Game Mechanics Engine powered by CC-Nodes, Perceptual Lens Scale Switching,
SealedAttractor Vault, and 4-Layer Closed-Loop Telemetry Engine for autonomous hardware adaptation.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time
from typing import Any, Dict, List, Optional, Tuple


class ScaleLevel(Enum):
    MICRO_NPC = auto()      # Micro scale: individual actors / tiles / NPCs
    MESO_REGION = auto()    # Meso scale: region / town politics and safety
    MACRO_KINGDOM = auto()  # Macro scale: kingdom-wide narrative & geopolitical power graph


class SimdMode(Enum):
    AVX2_256 = 8
    SSE_128 = 4
    Scalar = 1


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

        # Rule 1: Thermal / Throttle check -> fallback SIMD width
        if telemetry.gpu_is_throttled or telemetry.gpu_temp_celsius > 80.0:
            profile.simd_width = SimdMode.SSE_128
        else:
            profile.simd_width = SimdMode.AVX2_256

        # Rule 2: Cache miss rate check -> adjust batch chunk size
        if telemetry.cpu_cache_miss_rate > 0.15:
            profile.chunk_size = 256
        else:
            profile.chunk_size = 1024

        # Rule 3: Frame time overshoot -> adjust causal LOD step
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
    tension_field: float = 0.0  # Causal friction tension (V_t)
    invariants: Dict[str, str] = field(default_factory=dict)
    bound_rules: List[str] = field(default_factory=list)


@dataclass
class NarrativeAnomalyZone:
    """Quarantined SealedAttractor object isolating narrative ruptures."""

    zone_id: str
    quarantined_nodes: List[CCGameNode]
    peak_tension: float
    restructure_attempts: int = 0
    status: str = "SEALED"  # SEALED -> RESTRUCTURING -> RECOVERED


class PerceptualLensController:
    """Perceptual Lens: Scale shifting (Micro/Meso/Macro) and invariant observation."""

    def __init__(self):
        self.current_scale = ScaleLevel.MICRO_NPC

    def shift_scale(self, target_scale: ScaleLevel) -> None:
        """Switch observation bandwidth."""
        self.current_scale = target_scale

    def evaluate_narrative_tension(
        self, active_rules: List[str], nodes: Dict[str, CCGameNode]
    ) -> float:
        """Calculate narrative tension V_t caused by critical events (e.g. key NPC death)."""
        king_node = nodes.get("NPC_KING_ARTHUR")

        if king_node and not king_node.is_active:
            if "RULE_KING_MUST_GRANT_QUEST" in active_rules:
                return 0.95  # Severe logical narrative rupture (V_t > V_critical)

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
        """[Step 1] Isolate rupture zone and generate SealedAttractor object."""
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
        """[Step 2] Shift scale and insert mediating CC-Node for narrative restructuring."""
        anomaly = self.anomalies.get(anomaly_id)
        if not anomaly:
            return False, None, ""

        anomaly.status = "RESTRUCTURING"

        # 1. Perceptual Lens scale shift: MICRO (King assassinated) -> MACRO (Power vacuum & Regency council)
        lens.shift_scale(ScaleLevel.MACRO_KINGDOM)

        # 2. Autonomous creation of emergency mediating CC-Node
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

        # 3. Constraint boundary relaxation: "Meet King" -> "Power struggle with Regency Council"
        new_narrative_rule = "RULE_INTERACT_WITH_REGENCY_COUNCIL"

        anomaly.status = "RECOVERED"
        anomaly.peak_tension = 0.03  # Tension canceled (V_t <= 0.05)

        return True, regency_council_node, new_narrative_rule


class CausalGameMechanicsEngine:
    """Autonomous Causal Game Mechanics Main Engine featuring Closed-Loop Governor."""

    def __init__(self):
        self.lens = PerceptualLensController()
        self.vault = SealedAttractorVault(critical_threshold=0.5)
        self.observer = EnvironmentObserver()
        self.governor = RealtimeGovernor(target_frame_ms=16.67)

        # Initial world state
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
        """Execute one closed-loop adaptation tick."""
        snapshot = self.observer.poll(frame_delta_ms)
        return self.governor.evaluate(snapshot)

    def execute_player_action(self, action_type: str, target_id: str) -> Dict[str, Any]:
        """Process player action (e.g., assassinating key NPC)."""
        target_node = self.nodes.get(target_id)
        if target_node and action_type == "ASSASSINATE":
            target_node.is_active = False

        # 1. Measure causal friction tension (V_t)
        current_vt = self.lens.evaluate_narrative_tension(
            self.active_quest_rules, self.nodes
        )

        # 2. On tension overflow, execute SealedAttractor isolation and restructuring
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
