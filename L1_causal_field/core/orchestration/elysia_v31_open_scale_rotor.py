"""
ELYSIA v3.1: Hierarchical Open Scale-Rotor Interactive Causal Architecture
==========================================================================

Implements the ELYSIA v3.1 Master Design Specification:
1. Local Closed Droplets ("물방울", `CausalDroplet`):
   Respects each bounded local world's internal repulsion, surface tension, and
   bonding arrangement as valid *within its own scale*, without falsely reducing
   the outer macro world into one droplet's narrow rule.
2. Self-Structure Hierarchical Discernment (`CausalDroplet.discern_against_self`):
   Zero external type labels ("atom"/"molecule"). A composite structure uses its
   OWN internal bond topology, constituent count, surface tension, and scale
   radius as the discriminant coordinate to classify encountered structures as
   `SUB_TIER_CONSTITUENT`, `PEER_ELEMENTARY`, `PEER_TIER_COMPOSITE`, or
   `SUPER_TIER_ENCLOSURE`, and bond upward into macro hierarchies.
3. Open Spherical Virtual Relational Memory (`OpenVirtualRelationalMemory`):
   Embeds spontaneously sprouted `CognitiveAxis` & `CausalDroplet` entities onto
   concentric spherical shells using `ScaleRotor` (`core.topology.causal_scale_rotor`)
   woven with orthogonal `warp` (vertical cross-scale) and `weft` (horizontal
   cross-modality) links, plus spherical mirror phase-gradient inference when
   any node is occluded.
4. Interactive Learning Protocol (`InteractionProtocol`):
   Zero pre-labeled puzzles or hardcoded target answers. Actively perturbs
   external stimuli, detects continuous response correlation spectra & invariant
   channels, spontaneously births `CognitiveAxis` structures, assesses which
   physical sensing modality remains viable (e.g., dense fog optical blindness
   vs EM radar self-selection without hardcoded `if fog:` rules), and absorbs
   unexpected exceptions as `BoundaryFrictionEvent` nodes without crashing.
5. Real External World Bridge (`RealExternalToolBridge`):
   Executes real OS Python subprocesses, reads real workspace `.py` ASTs and
   real media (`real_human_face_hd.jpg`, `unified_creation_dawn.wav`), and
   absorbs real external correlations & failure boundaries onto the sphere.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings
import wave

import numpy as np
from PIL import Image

from core.topology.causal_scale_rotor import ScaleRotor

if TYPE_CHECKING:
    from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
    from core.physics.causal_field import CausalField


# ---------------------------------------------------------------------------
# 1. Local Closed Droplet ("물방울") & Self-Structure Hierarchy Discernment
# ---------------------------------------------------------------------------


class HierarchyRelation(str, Enum):
    """Structural tier relation discerned strictly relative to Self's own topology."""

    SUB_TIER_CONSTITUENT = "sub_tier_constituent"
    PEER_ELEMENTARY = "peer_elementary"
    PEER_TIER_COMPOSITE = "peer_tier_composite"
    SUPER_TIER_ENCLOSURE = "super_tier_enclosure"


@dataclass
class HierarchyDiscernment:
    """Result of one droplet measuring another using ONLY its own internal structure."""

    observer_id: str
    target_id: str
    relation: HierarchyRelation
    scale_ratio: float
    bond_complexity_ratio: float
    spectral_resonance: float
    can_bond_higher: bool
    reasoning_ko: str


@dataclass
class CausalDroplet:
    """
    A bounded local causal world ("물방울") held together by internal repulsion,
    surface tension, and bonding topology. Carries NO external tier label.
    """

    droplet_id: str
    constituents: Tuple[str, ...] = ()
    internal_bonds: Tuple[Tuple[int, int, float], ...] = ()
    repulsion_balance: float = 0.5
    surface_tension: float = 0.5
    scale_radius: float = 1.0
    spectral_signature: np.ndarray = field(
        default_factory=lambda: np.ones(8, dtype=np.float64) / math.sqrt(8.0)
    )
    open_valences: int = 2
    local_order_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vec = np.asarray(self.spectral_signature, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(vec))
        self.spectral_signature = (vec / norm) if norm > 1e-12 else (np.ones(8, dtype=np.float64) / math.sqrt(8.0))

    def structural_signature(self) -> Dict[str, float]:
        n_const = max(1, len(self.constituents))
        n_bonds = len(self.internal_bonds)
        return {
            "constituent_count": float(len(self.constituents)),
            "bond_count": float(n_bonds),
            "bond_density": float(n_bonds / n_const),
            "surface_tension": float(self.surface_tension),
            "repulsion_balance": float(self.repulsion_balance),
            "scale_radius": float(self.scale_radius),
        }

    def discern_against_self(self, other: "CausalDroplet") -> HierarchyDiscernment:
        """
        Discriminates `other`'s structural tier using ONLY `self`'s own bonding
        and spatial scale arrangement as the coordinate benchmark.
        """
        s_bonds = len(self.internal_bonds)
        o_bonds = len(other.internal_bonds)
        s_const = len(self.constituents)
        o_const = len(other.constituents)
        scale_ratio = float(other.scale_radius / max(1e-6, self.scale_radius))
        bond_ratio = float((o_bonds + 1.0) / (s_bonds + 1.0))

        # Align spectral dimensions cleanly if lengths differ
        n_min = min(len(self.spectral_signature), len(other.spectral_signature))
        spec_res = float(
            np.dot(self.spectral_signature[:n_min], other.spectral_signature[:n_min])
        )
        can_bond = bool(self.open_valences > 0 and other.open_valences > 0)

        if s_bonds == 0 and o_bonds == 0 and 0.4 <= scale_ratio <= 2.5:
            rel = HierarchyRelation.PEER_ELEMENTARY
            reason = (
                f"내 구조(결합=0, 반경={self.scale_radius:.2f})를 잣대로 보니 대상 '{other.droplet_id}' 역시 "
                f"내부 결합이 없는 동일 단일 단위(하위 원소) 계층입니다."
            )
        elif (
            self.droplet_id in other.constituents
            or scale_ratio > 2.0
            or (o_const >= max(2, 2 * max(1, s_const)) and o_bonds > s_bonds)
        ):
            rel = HierarchyRelation.SUPER_TIER_ENCLOSURE
            reason = (
                f"내 구조(결합={s_bonds}, 반경={self.scale_radius:.2f})보다 대상 '{other.droplet_id}'의 "
                f"포괄 반경({other.scale_radius:.2f}, 비={scale_ratio:.2f})·내부 결합({o_bonds})이 상위로 확장되어 있어 "
                "나를 감싸는 상위 거시 계층으로 분별됩니다."
            )
        elif o_bonds == 0 and s_bonds > 0 and scale_ratio < 0.95:
            rel = HierarchyRelation.SUB_TIER_CONSTITUENT
            reason = (
                f"나는 내부 결합({s_bonds}개)으로 짜인 복합체이지만 대상 '{other.droplet_id}'은 "
                f"단일 미결합 단위(반경 비={scale_ratio:.2f})이므로 내 하위 구성(원자) 계층으로 분별됩니다."
            )
        elif o_bonds > 0 and s_bonds > 0 and 0.45 <= scale_ratio <= 2.0:
            rel = HierarchyRelation.PEER_TIER_COMPOSITE
            reason = (
                f"대상 '{other.droplet_id}'은 나(결합={s_bonds}, 반경={self.scale_radius:.2f})와 "
                f"대등한 내부 결합({o_bonds})과 표면장력({other.surface_tension:.2f})을 이룬 동일 복합(분자) 계층입니다."
            )
        elif scale_ratio < 1.0:
            rel = HierarchyRelation.SUB_TIER_CONSTITUENT
            reason = f"내 스케일 반경 기준(비={scale_ratio:.2f})으로 하위 계층 단위로 분별됩니다."
        else:
            rel = HierarchyRelation.SUPER_TIER_ENCLOSURE
            reason = f"내 스케일 반경 기준(비={scale_ratio:.2f})으로 상위 거시 계층으로 분별됩니다."

        return HierarchyDiscernment(
            observer_id=self.droplet_id,
            target_id=other.droplet_id,
            relation=rel,
            scale_ratio=scale_ratio,
            bond_complexity_ratio=bond_ratio,
            spectral_resonance=spec_res,
            can_bond_higher=can_bond,
            reasoning_ko=reason,
        )

    def bond_with(
        self,
        other: "CausalDroplet",
        new_id: str,
        bond_strength: float = 0.85,
    ) -> "CausalDroplet":
        """Forms an emergent higher-scale droplet while preserving local child orders."""
        children = (self.droplet_id, other.droplet_id)
        # Preserve internal bonds from both children + add emergent inter-droplet bond
        offset = max(1, len(self.constituents) or 1)
        merged_bonds: List[Tuple[int, int, float]] = list(self.internal_bonds)
        for i, j, w in other.internal_bonds:
            merged_bonds.append((i + offset, j + offset, w))
        merged_bonds.append((0, offset, float(bond_strength)))

        new_radius = float(math.hypot(self.scale_radius, other.scale_radius) * 1.35)
        new_tension = float(min(1.0, 0.5 * (self.surface_tension + other.surface_tension) + 0.15 * bond_strength))
        new_rep = float(0.5 * (self.repulsion_balance + other.repulsion_balance))

        n_len = max(len(self.spectral_signature), len(other.spectral_signature))
        s1 = np.pad(self.spectral_signature, (0, n_len - len(self.spectral_signature)))
        s2 = np.pad(other.spectral_signature, (0, n_len - len(other.spectral_signature)))
        merged_sig = s1 + bond_strength * s2

        new_valences = max(1, (self.open_valences + other.open_valences) - 2)
        return CausalDroplet(
            droplet_id=new_id,
            constituents=children,
            internal_bonds=tuple(merged_bonds),
            repulsion_balance=new_rep,
            surface_tension=new_tension,
            scale_radius=new_radius,
            spectral_signature=merged_sig,
            open_valences=new_valences,
            local_order_notes={
                "emerged_from": list(children),
                "preserved_child_radii": [self.scale_radius, other.scale_radius],
            },
        )


# ---------------------------------------------------------------------------
# 2. Open Spherical Virtual Relational Memory & Scale-Invariant Rotor
# ---------------------------------------------------------------------------


@dataclass
class CognitiveAxis:
    """A spontaneously formed cognitive axis grounded in lived stimulus-response spectra."""

    axis_id: str
    stimulus_channel: str
    response_channels: Tuple[str, ...]
    correlation_spectrum: Dict[str, float]
    invariant_channels: Tuple[str, ...]
    dominant_correlation: float
    dominant_slope: float
    monotonicity: float
    sensitivity_resolution: float
    scale_tier: str
    scale_radius: float
    azimuth_theta: float
    elevation_phi: float
    rotor: ScaleRotor
    sample_count: int
    origin_medium: str


@dataclass
class WeaveLink:
    """Curved-surface warp (cross-scale) or weft (cross-modality) relational link."""

    src_id: str
    dst_id: str
    fabric_kind: str  # "warp" | "weft"
    coupling_strength: float
    geodesic_angle_rad: float


@dataclass
class BoundaryFrictionEvent:
    """Non-crashing absorption of unexpected external errors, noise, or sensory occlusion."""

    event_id: str
    source_name: str
    stimulus_snapshot: Dict[str, Any]
    friction_kind: str
    detail: str


@dataclass
class SphericalNode:
    """Entity placed on the open concentric spherical relational memory manifold."""

    node_id: str
    node_kind: str  # "cognitive_axis" | "causal_droplet" | "boundary_horizon"
    scale_tier: str
    scale_radius: float
    azimuth_theta: float
    elevation_phi: float
    rotor: ScaleRotor
    signature_vec: np.ndarray
    live_value: float = 0.0
    payload: Any = None

    def unit_vec(self) -> np.ndarray:
        return np.array(
            [
                math.sin(self.elevation_phi) * math.cos(self.azimuth_theta),
                math.sin(self.elevation_phi) * math.sin(self.azimuth_theta),
                math.cos(self.elevation_phi),
            ],
            dtype=np.float64,
        )

    def position_3d(self) -> np.ndarray:
        return self.scale_radius * self.unit_vec()


def _tier_from_radius(r: float) -> str:
    if r < 1.25:
        return "earth"
    if r < 2.85:
        return "water"
    return "sky"


class OpenVirtualRelationalMemory:
    """
    Spherical curved-surface relational memory organizing axes, droplets, and
    external boundary discoveries with orthogonal `warp` (cross-scale) & `weft`
    (cross-axis) weaving plus geodesic mirror inference for occluded nodes.
    """

    _GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

    def __init__(self) -> None:
        self.nodes: Dict[str, SphericalNode] = {}
        self.axes: Dict[str, CognitiveAxis] = {}
        self.droplets: Dict[str, CausalDroplet] = {}
        self.boundary_events: List[BoundaryFrictionEvent] = []
        self.weave_links: List[WeaveLink] = []
        self._link_keys: set[Tuple[str, str, str]] = set()

    def _allocate_spherical_angles(self) -> Tuple[float, float]:
        idx = len(self.nodes) + 1
        theta = float((idx * self._GOLDEN_ANGLE) % (2.0 * math.pi))
        # Spread elevation smoothly across (0.25*pi .. 0.75*pi)
        frac = ((idx * 0.61803398875) % 1.0)
        phi = float(math.acos(max(-0.92, min(0.92, 1.0 - 2.0 * frac))))
        return theta, phi

    @staticmethod
    def _geodesic_angle(a: SphericalNode, b: SphericalNode) -> float:
        dot = float(np.clip(np.dot(a.unit_vec(), b.unit_vec()), -1.0, 1.0))
        return float(math.acos(dot))

    def _add_link(self, a_id: str, b_id: str, fabric_kind: str, strength: float) -> None:
        if a_id == b_id or a_id not in self.nodes or b_id not in self.nodes:
            return
        key = (min(a_id, b_id), max(a_id, b_id), fabric_kind)
        if key in self._link_keys:
            return
        self._link_keys.add(key)
        ang = self._geodesic_angle(self.nodes[a_id], self.nodes[b_id])
        self.weave_links.append(
            WeaveLink(
                src_id=a_id,
                dst_id=b_id,
                fabric_kind=fabric_kind,
                coupling_strength=float(max(0.05, min(1.0, strength))),
                geodesic_angle_rad=ang,
            )
        )

    def _auto_weave_node(self, new_node: SphericalNode, explicit_warp_parents: Sequence[str] = ()) -> None:
        for pid in explicit_warp_parents:
            if pid in self.nodes:
                self._add_link(pid, new_node.node_id, "warp", 0.92)

        for other_id, other in self.nodes.items():
            if other_id == new_node.node_id:
                continue
            n_min = min(len(new_node.signature_vec), len(other.signature_vec))
            v1 = new_node.signature_vec[:n_min]
            v2 = other.signature_vec[:n_min]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            sim = float(abs(np.dot(v1, v2)) / denom) if denom > 1e-12 else 0.0

            if other.scale_tier == new_node.scale_tier:
                # Same spherical shell -> horizontal WEFT weaving
                if sim >= 0.25 or len(self.nodes) <= 6:
                    self._add_link(other_id, new_node.node_id, "weft", max(0.25, sim))
            else:
                # Different concentric shell -> vertical WARP scale-invariant weaving
                if sim >= 0.35:
                    self._add_link(other_id, new_node.node_id, "warp", max(0.30, sim))

    def absorb_axis(self, axis: CognitiveAxis) -> SphericalNode:
        self.axes[axis.axis_id] = axis
        corr_vals = np.array(list(axis.correlation_spectrum.values()), dtype=np.float64)
        if len(corr_vals) < 8:
            sig = np.pad(corr_vals, (0, 8 - len(corr_vals)))
        else:
            sig = corr_vals[:8]
        norm = float(np.linalg.norm(sig))
        sig = (sig / norm) if norm > 1e-12 else (np.ones(8, dtype=np.float64) / math.sqrt(8.0))

        node = SphericalNode(
            node_id=axis.axis_id,
            node_kind="cognitive_axis",
            scale_tier=axis.scale_tier,
            scale_radius=axis.scale_radius,
            azimuth_theta=axis.azimuth_theta,
            elevation_phi=axis.elevation_phi,
            rotor=axis.rotor,
            signature_vec=sig,
            live_value=float(axis.dominant_correlation),
            payload=axis,
        )
        self.nodes[node.node_id] = node
        self._auto_weave_node(node)
        return node

    def absorb_droplet(
        self,
        droplet: CausalDroplet,
        explicit_parents: Sequence[str] = (),
    ) -> SphericalNode:
        self.droplets[droplet.droplet_id] = droplet
        theta, phi = self._allocate_spherical_angles()
        tier = _tier_from_radius(droplet.scale_radius)
        rotor = ScaleRotor(
            axis=(math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)),
            angular_rate=1.0 / max(0.3, droplet.scale_radius),
            phase=theta,
            radius=droplet.scale_radius,
            scale=droplet.surface_tension,
        )
        node = SphericalNode(
            node_id=droplet.droplet_id,
            node_kind="causal_droplet",
            scale_tier=tier,
            scale_radius=droplet.scale_radius,
            azimuth_theta=theta,
            elevation_phi=phi,
            rotor=rotor,
            signature_vec=droplet.spectral_signature.copy(),
            live_value=float(droplet.surface_tension),
            payload=droplet,
        )
        self.nodes[node.node_id] = node
        warp_parents = list(dict.fromkeys([*droplet.constituents, *explicit_parents]))
        self._auto_weave_node(node, explicit_warp_parents=warp_parents)
        return node

    def record_boundary_friction(self, event: BoundaryFrictionEvent) -> SphericalNode:
        self.boundary_events.append(event)
        theta, phi = self._allocate_spherical_angles()
        r_sky = 4.2
        sig = np.zeros(8, dtype=np.float64)
        sig[0] = 0.7
        sig[7] = 0.7
        rotor = ScaleRotor(axis=(0.0, 0.0, 1.0), angular_rate=0.5, phase=theta, radius=r_sky, scale=1.0)
        node = SphericalNode(
            node_id=event.event_id,
            node_kind="boundary_horizon",
            scale_tier="sky",
            scale_radius=r_sky,
            azimuth_theta=theta,
            elevation_phi=phi,
            rotor=rotor,
            signature_vec=sig,
            live_value=-1.0,
            payload=event,
        )
        self.nodes[node.node_id] = node
        self._auto_weave_node(node)
        return node

    def infer_missing_node_value(
        self,
        occluded_id: str,
        live_values: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Reconstructs an occluded/dropped node's state from surrounding warp & weft
        spherical mirror neighbors via geodesic angular weighting (zero crash).
        """
        if occluded_id not in self.nodes:
            return {"inferred": False, "estimated_value": None, "confidence": 0.0, "neighbors_used": []}

        weighted_sum = 0.0
        total_w = 0.0
        used: List[str] = []

        for link in self.weave_links:
            nbr: Optional[str] = None
            if link.src_id == occluded_id and link.dst_id in live_values:
                nbr = link.dst_id
            elif link.dst_id == occluded_id and link.src_id in live_values:
                nbr = link.src_id
            if nbr is None:
                continue
            w = float(link.coupling_strength * math.exp(-0.5 * link.geodesic_angle_rad))
            weighted_sum += w * float(live_values[nbr])
            total_w += w
            used.append(nbr)

        if total_w <= 1e-12:
            return {"inferred": False, "estimated_value": None, "confidence": 0.0, "neighbors_used": []}

        est = float(weighted_sum / total_w)
        conf = float(1.0 - math.exp(-total_w))
        return {
            "inferred": True,
            "estimated_value": est,
            "confidence": conf,
            "neighbors_used": sorted(set(used)),
        }

    def topology_metrics(self) -> Dict[str, Any]:
        n_nodes = len(self.nodes)
        n_warp = sum(1 for l in self.weave_links if l.fabric_kind == "warp")
        n_weft = sum(1 for l in self.weave_links if l.fabric_kind == "weft")
        max_edges = max(1, (n_nodes * (n_nodes - 1)) // 2)
        tiers = sorted({n.scale_tier for n in self.nodes.values()})
        return {
            "node_count": n_nodes,
            "axis_count": len(self.axes),
            "droplet_count": len(self.droplets),
            "boundary_event_count": len(self.boundary_events),
            "warp_links": n_warp,
            "weft_links": n_weft,
            "weave_density": float(len(self.weave_links) / max_edges),
            "scale_tiers_occupied": tiers,
        }

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "v31",
    ) -> Dict[str, Any]:
        """Projects spherical nodes and warp/weft links into `CausalField` (strict 4D tensors)."""
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        vids: Dict[str, str] = {}
        for nid, node in self.nodes.items():
            vid = f"{prefix}_{nid}"
            vids[nid] = vid
            pos = node.position_3d()[: field.dimensions]
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"node_id": nid, "kind": node.node_kind, "tier": node.scale_tier},
                    tensor=np.array(
                        [
                            node.scale_radius / 5.0,
                            node.azimuth_theta / (2.0 * math.pi),
                            node.elevation_phi / math.pi,
                            float(math.tanh(node.live_value)),
                        ],
                        dtype=np.float32,
                    ),
                    mass=max(0.5, float(node.scale_radius)),
                    position=np.asarray(pos, dtype=np.float32),
                    phase_angle=float(node.azimuth_theta),
                    scale_level=float(node.scale_radius),
                    curiosity_charge=1.0 if node.node_kind == "boundary_horizon" else 0.2,
                )
            )

        linked = 0
        for link in self.weave_links:
            if link.src_id in vids and link.dst_id in vids:
                field.link_voxels(vids[link.src_id], vids[link.dst_id], strength=link.coupling_strength)
                linked += 1

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_open_sphere",
                name="ElysiaV31OpenScaleRotorSphere",
                position=np.zeros(field.dimensions, dtype=np.float32),
                intensity=2.8,
                tier="macro",
                symbolic_intent=f"Nodes={len(self.nodes)}, Warp+Weft={len(self.weave_links)}",
            )
        )
        return {
            "causal_field": field,
            "voxels": vids,
            "beams_linked": linked,
        }


# ---------------------------------------------------------------------------
# 3. Interactive Learning Protocol & Modality Viability Self-Assessment
# ---------------------------------------------------------------------------


def _rank_array(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(xc, yc) / denom, -1.0, 1.0))


@dataclass
class ModalityAssessmentReport:
    """Self-assessment of which sensing field retains usable resolution under live conditions."""

    stimulus_channel: str
    modality_scores: Dict[str, Dict[str, float]]
    viable_modalities: List[str]
    unusable_modalities: List[str]
    selected_modality: Optional[str]
    formed_axis: Optional[CognitiveAxis]
    reasoning_ko: str


class InteractionProtocol:
    """
    Open interactive learning protocol (ELYSIA v3.1 §2.1 & §3.2):
    Perturbs external media, senses correlation spectra across raw channels,
    spontaneously births `CognitiveAxis` structures, selects viable sensory
    modalities empirically without hardcoded rules, and never crashes on friction.
    """

    def __init__(self, memory: OpenVirtualRelationalMemory) -> None:
        self.memory = memory
        self.interaction_count: int = 0

    def interact_and_discover_axis(
        self,
        medium_name: str,
        interact_fn: Callable[[Dict[str, float]], Dict[str, Any]],
        stimulus_channel: str,
        sweep_values: Sequence[float],
        scale_tier: str = "earth",
        scale_radius: float = 1.0,
        correlation_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        stim_ok: List[float] = []
        channel_series: Dict[str, List[float]] = {}
        friction_events: List[BoundaryFrictionEvent] = []

        for val in sweep_values:
            self.interaction_count += 1
            stim = {stimulus_channel: float(val)}
            try:
                resp = interact_fn(stim)
                if not isinstance(resp, dict):
                    raise TypeError(f"Expected dict response, got {type(resp).__name__}")
                numeric_obs: Dict[str, float] = {}
                for k, v in resp.items():
                    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                        fv = float(v)
                        if math.isfinite(fv):
                            numeric_obs[str(k)] = fv
                if not numeric_obs:
                    raise ValueError("No finite numeric channels observed in external response")
            except Exception as exc:
                ev = BoundaryFrictionEvent(
                    event_id=f"boundary::{medium_name}::{self.interaction_count}",
                    source_name=medium_name,
                    stimulus_snapshot=stim,
                    friction_kind=type(exc).__name__,
                    detail=str(exc),
                )
                self.memory.record_boundary_friction(ev)
                friction_events.append(ev)
                continue

            stim_ok.append(float(val))
            for k, fv in numeric_obs.items():
                channel_series.setdefault(k, []).append(fv)

        if len(stim_ok) < 3:
            return {
                "axis": None,
                "correlation_spectrum": {},
                "responsive_channels": [],
                "invariant_channels": [],
                "friction_events": friction_events,
                "sensitivity_resolution": 0.0,
                "summary_ko": (
                    f"매질 '{medium_name}'에서 유효 관측이 부족({len(stim_ok)}건, 경계 마찰 {len(friction_events)}건)해 "
                    "인지 축을 섣불리 단정하지 않고 경계 사건으로 흡수했습니다."
                ),
            }

        x = np.asarray(stim_ok, dtype=np.float64)
        x_span = max(1e-9, float(np.max(x) - np.min(x)))
        x_ranks = _rank_array(x)

        corr_spec: Dict[str, float] = {}
        mono_spec: Dict[str, float] = {}
        slope_spec: Dict[str, float] = {}
        res_spec: Dict[str, float] = {}

        for ch, vals in channel_series.items():
            if len(vals) != len(stim_ok):
                continue
            y = np.asarray(vals, dtype=np.float64)
            y_std = float(np.std(y))
            r = _safe_pearson(x, y) if y_std > 1e-6 else 0.0
            m = _safe_pearson(x_ranks, _rank_array(y)) if y_std > 1e-6 else 0.0
            slope = float(np.cov(x, y, bias=True)[0, 1] / max(1e-12, float(np.var(x)))) if y_std > 1e-6 else 0.0
            # Signal clarity (resolution): amplitude swing normalized by baseline scale
            swing = float(np.max(y) - np.min(y))
            res = float(abs(r) * math.tanh(swing / max(0.05, 0.25 * (abs(float(np.mean(y))) + 1.0))))
            corr_spec[ch] = r
            mono_spec[ch] = m
            slope_spec[ch] = slope
            res_spec[ch] = res

        responsive = [
            ch for ch, r in corr_spec.items() if abs(r) >= correlation_threshold and res_spec.get(ch, 0.0) >= 0.08
        ]
        invariant = [
            ch for ch, r in corr_spec.items() if abs(r) < 0.25 or res_spec.get(ch, 0.0) < 0.08
        ]

        if not responsive:
            max_res = max(res_spec.values()) if res_spec else 0.0
            return {
                "axis": None,
                "correlation_spectrum": corr_spec,
                "responsive_channels": [],
                "invariant_channels": sorted(invariant),
                "friction_events": friction_events,
                "sensitivity_resolution": float(max_res),
                "summary_ko": (
                    f"매질 '{medium_name}'의 자극 '{stimulus_channel}' 변화에 뚜렷하게 상관 반응하는 감각 축을 "
                    f"포착하지 못했습니다(최대 해상도={max_res:.3f})."
                ),
            }

        responsive.sort(key=lambda c: (-abs(corr_spec[c]), -res_spec[c]))
        top_ch = responsive[0]
        theta, phi = self.memory._allocate_spherical_angles()
        rotor = ScaleRotor(
            axis=(math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)),
            angular_rate=abs(corr_spec[top_ch]),
            phase=theta,
            radius=scale_radius,
            scale=max(0.1, res_spec[top_ch]),
        )
        axis_id = f"axis::{medium_name}::{stimulus_channel}_to_{top_ch}"
        axis = CognitiveAxis(
            axis_id=axis_id,
            stimulus_channel=stimulus_channel,
            response_channels=tuple(responsive),
            correlation_spectrum=corr_spec,
            invariant_channels=tuple(sorted(invariant)),
            dominant_correlation=float(corr_spec[top_ch]),
            dominant_slope=float(slope_spec[top_ch]),
            monotonicity=float(mono_spec[top_ch]),
            sensitivity_resolution=float(res_spec[top_ch]),
            scale_tier=scale_tier,
            scale_radius=scale_radius,
            azimuth_theta=theta,
            elevation_phi=phi,
            rotor=rotor,
            sample_count=len(stim_ok),
            origin_medium=medium_name,
        )
        self.memory.absorb_axis(axis)

        summary = (
            f"사전 라벨 없이 '{medium_name}'에서 '{stimulus_channel}'(폭 {x_span:.2g})을 변화시키며 재감각한 결과, "
            f"불변 채널({', '.join(sorted(invariant)) or '없음'})과 분리되어 반응 채널({', '.join(responsive)})이 "
            f"상관계수 r={corr_spec[top_ch]:+.3f}(단조성 {mono_spec[top_ch]:+.3f}, 해상도 {res_spec[top_ch]:.3f})으로 "
            f"공명함을 지각해 새 인지 축 '{axis_id}'를 자생했습니다."
        )
        return {
            "axis": axis,
            "correlation_spectrum": corr_spec,
            "responsive_channels": responsive,
            "invariant_channels": sorted(invariant),
            "friction_events": friction_events,
            "sensitivity_resolution": float(res_spec[top_ch]),
            "summary_ko": summary,
        }

    def assess_and_select_viable_modality(
        self,
        modalities: Dict[str, Callable[[Dict[str, float]], Dict[str, Any]]],
        stimulus_channel: str,
        sweep_values: Sequence[float],
        min_viable_resolution: float = 0.25,
    ) -> ModalityAssessmentReport:
        """
        Empirical self-assessment across physical sensing fields (e.g., dense fog
        degrading optical field while EM radar preserves correlation) with ZERO
        hardcoded environment rules.
        """
        scores: Dict[str, Dict[str, float]] = {}
        viable: List[str] = []
        unusable: List[str] = []
        axes_by_mod: Dict[str, Optional[CognitiveAxis]] = {}

        for mod_name, fn in modalities.items():
            res = self.interact_and_discover_axis(
                medium_name=mod_name,
                interact_fn=fn,
                stimulus_channel=stimulus_channel,
                sweep_values=sweep_values,
                scale_tier="water",
                scale_radius=2.2,
            )
            ax = res["axis"]
            sens = float(res["sensitivity_resolution"])
            max_abs_r = max((abs(v) for v in res["correlation_spectrum"].values()), default=0.0)
            scores[mod_name] = {
                "max_abs_correlation": float(max_abs_r),
                "sensitivity_resolution": sens,
            }
            axes_by_mod[mod_name] = ax
            if ax is not None and sens >= min_viable_resolution:
                viable.append(mod_name)
            else:
                unusable.append(mod_name)
                self.memory.record_boundary_friction(
                    BoundaryFrictionEvent(
                        event_id=f"boundary::modality_blind::{mod_name}_{self.interaction_count}",
                        source_name=mod_name,
                        stimulus_snapshot={"channel": stimulus_channel},
                        friction_kind="SensoryResolutionCollapse",
                        detail=f"Resolution={sens:.3f}, max|r|={max_abs_r:.3f} insufficient for discernment",
                    )
                )

        selected: Optional[str] = None
        if viable:
            selected = max(
                viable,
                key=lambda m: (
                    scores[m]["sensitivity_resolution"],
                    scores[m]["max_abs_correlation"],
                ),
            )
            reason = (
                f"가용 감각·필드 {len(modalities)}종을 직접 탐사해 본 결과, "
                f"감지 해상도가 붕괴한 채널({', '.join(unusable) or '없음'})을 스스로 한계로 판별하고 "
                f"상관 스펙트럼이 선명히 살아있는 '{selected}'(해상도={scores[selected]['sensitivity_resolution']:.3f}, "
                f"|r|={scores[selected]['max_abs_correlation']:.3f})를 능동 선택했습니다."
            )
        else:
            reason = (
                f"탐사한 모든 감각 채널({', '.join(unusable)})에서 유효 해상도를 얻지 못해 "
                "현재 도구 한계를 자각하고 새로운 외부 물리 매질 탐색이 필요합니다."
            )

        return ModalityAssessmentReport(
            stimulus_channel=stimulus_channel,
            modality_scores=scores,
            viable_modalities=viable,
            unusable_modalities=unusable,
            selected_modality=selected,
            formed_axis=axes_by_mod.get(selected) if selected is not None else None,
            reasoning_ko=reason,
        )


# ---------------------------------------------------------------------------
# 4. Label-Free External Physical Media & Spatial Object Field (Scenarios 1 & 2)
# ---------------------------------------------------------------------------


class OpaqueSensoryMedium:
    """
    Scenario 1 external medium (Chili/Spiciness principle with ZERO labels):
    Exposes 5 unlabeled sensory channels where only `receptor_ch3` responds
    monotonically to `substance_dose`, while the other 4 channels remain invariant.
    """

    def __init__(self, seed: int = 11) -> None:
        self._rng = np.random.default_rng(seed)

    def interact(self, stimulus: Dict[str, float]) -> Dict[str, float]:
        dose = max(0.0, float(stimulus["substance_dose"]))
        noise = self._rng.normal(0.0, 0.015, size=5)
        return {
            "receptor_ch0": float(1.20 + noise[0]),
            "receptor_ch1": float(0.85 + noise[1]),
            "receptor_ch2": float(1.50 + noise[2]),
            "receptor_ch3": float((10.0 * dose) / (2.0 + dose) + noise[3]),
            "receptor_ch4": float(0.60 + noise[4]),
        }


class FoggyRoadMultiFieldEnvironment:
    """
    External physical environment testing self-assessment of sensory capability:
    Dense fog scatters visible optics (destroying distance correlation) while
    long-wave EM radar penetrates cleanly.
    """

    def __init__(self, fog_density: float = 0.92, seed: int = 23) -> None:
        self.fog_density = float(np.clip(fog_density, 0.0, 1.0))
        self._rng = np.random.default_rng(seed)

    def probe_optical(self, stimulus: Dict[str, float]) -> Dict[str, float]:
        dist = max(1.0, float(stimulus["obstacle_distance_m"]))
        # Beer-Lambert optical extinction in dense radiation fog (sigma ~ 0.65 * fog_density m^-1)
        trans = math.exp(-0.65 * self.fog_density * dist)
        clear_contrast = 1.0 / (1.0 + 0.08 * dist)
        scatter_floor = 0.85 * self.fog_density
        noise_sigma = 0.008 + 0.035 * self.fog_density
        obs_contrast = trans * clear_contrast + (1.0 - trans) * 0.01 + float(self._rng.normal(0.0, noise_sigma))
        obs_luminance = trans * clear_contrast + scatter_floor + float(self._rng.normal(0.0, noise_sigma))
        return {
            "optical_edge_contrast": float(max(0.0, obs_contrast)),
            "optical_field_luminance": float(obs_luminance),
        }

    def probe_em_radar(self, stimulus: Dict[str, float]) -> Dict[str, float]:
        dist = max(1.0, float(stimulus["obstacle_distance_m"]))
        # Long-wave EM propagates cleanly through water-droplet fog
        echo_delay_ns = (2.0 * dist) / 0.299792
        echo_amplitude = 12.0 / (1.0 + 0.04 * dist)
        return {
            "em_echo_delay_ns": float(echo_delay_ns + self._rng.normal(0.0, 0.05)),
            "em_echo_amplitude": float(echo_amplitude + self._rng.normal(0.0, 0.01)),
        }


class UnlabeledSpatialObjectField:
    """
    Scenario 2A ("사과 원리"): An unlabeled physical object field exposing ONLY
    raw optical wavelength reflectance, spatial contour probing, and tactile
    indentation response — NO pre-attached name tag anywhere!
    """

    def __init__(self, radius_cm: float = 4.2, stiffness: float = 6.5) -> None:
        self._radius_cm = float(radius_cm)
        self._stiffness = float(stiffness)
        # 8-band reflectance (430nm violet -> 670nm deep red): strong long-wave reflection
        self._reflectance_8band = np.array(
            [0.08, 0.09, 0.11, 0.14, 0.25, 0.58, 0.86, 0.92], dtype=np.float64
        )

    def probe_contour(self, stimulus: Dict[str, float]) -> Dict[str, float]:
        ang = float(stimulus["probe_angle_rad"])
        # Smooth slightly dimpled convex closed contour
        r = self._radius_cm * (1.0 - 0.06 * math.cos(2.0 * ang))
        return {
            "boundary_radius_cm": float(r),
            "surface_normal_cos": float(math.cos(ang)),
            "long_wave_intensity": float(np.mean(self._reflectance_8band[5:])),
            "short_wave_intensity": float(np.mean(self._reflectance_8band[:3])),
        }

    def probe_tactile(self, stimulus: Dict[str, float]) -> Dict[str, float]:
        depth_mm = max(0.0, float(stimulus["indent_depth_mm"]))
        return {
            "restoring_force_n": float(self._stiffness * depth_mm),
            "surface_temperature_c": 21.0,
        }

    def optical_spectrum(self) -> np.ndarray:
        return self._reflectance_8band.copy()


def derive_structural_principle_label_free(
    protocol: InteractionProtocol,
    obj_field: UnlabeledSpatialObjectField,
) -> Dict[str, Any]:
    """
    Derives an entity's structural principle purely from probing its boundary,
    reflectance spectrum, and tactile resistance — with ZERO pre-given object label.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    radii: List[float] = []
    lw_vals: List[float] = []
    sw_vals: List[float] = []
    for a in angles:
        obs = obj_field.probe_contour({"probe_angle_rad": float(a)})
        radii.append(obs["boundary_radius_cm"])
        lw_vals.append(obs["long_wave_intensity"])
        sw_vals.append(obs["short_wave_intensity"])

    r_arr = np.asarray(radii, dtype=np.float64)
    mean_r = float(np.mean(r_arr))
    cv_r = float(np.std(r_arr) / max(1e-6, mean_r))
    closed_convex = bool(cv_r < 0.15 and float(np.min(r_arr)) > 0.0)
    long_short_ratio = float(np.mean(lw_vals) / max(1e-6, float(np.mean(sw_vals))))

    tact_res = protocol.interact_and_discover_axis(
        medium_name="spatial_object_tactile",
        interact_fn=obj_field.probe_tactile,
        stimulus_channel="indent_depth_mm",
        sweep_values=[0.1, 0.4, 0.8, 1.2, 1.8, 2.5],
        scale_tier="water",
        scale_radius=2.1,
    )
    firm_surface = bool(
        tact_res["axis"] is not None and tact_res["axis"].dominant_correlation > 0.95
    )

    droplet = CausalDroplet(
        droplet_id="principle::closed_convex_longwave_firm",
        constituents=("spatial_contour_probe", "optical_reflectance_probe", "tactile_resistance_probe"),
        internal_bonds=((0, 1, 0.90), (1, 2, 0.92), (0, 2, 0.88)),
        repulsion_balance=0.65,
        surface_tension=float(min(1.0, 0.5 + 0.4 * (1.0 - cv_r))),
        scale_radius=2.2,
        spectral_signature=obj_field.optical_spectrum(),
        open_valences=2,
        local_order_notes={
            "closed_convex_contour": closed_convex,
            "contour_regularity": round(1.0 - cv_r, 4),
            "long_to_short_wave_ratio": round(long_short_ratio, 3),
            "firm_surface_tension": firm_surface,
        },
    )
    protocol.memory.absorb_droplet(droplet)
    return {
        "droplet": droplet,
        "tactile_axis": tact_res["axis"],
        "closed_convex_contour": closed_convex,
        "long_to_short_wave_ratio": long_short_ratio,
        "firm_surface_tension": firm_surface,
        "summary_ko": (
            f"사전 라벨 없이 경계 곡률(정칙도={1.0 - cv_r:.3f})·광학 장파장 우세(비={long_short_ratio:.2f})·"
            f"촉각 복원 상관(firm={firm_surface})의 결합 작용만으로 고유 구조 원리 "
            f"'{droplet.droplet_id}'를 스스로 도출해 구면 관계망에 편입했습니다."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Real External Tool, Subprocess, Filesystem & Media Bridge (Scenario 3)
# ---------------------------------------------------------------------------


class RealExternalToolBridge:
    """
    Connects Elysia directly to real external OS Python execution, real workspace
    source files, and real sensory media files (`jpg`, `wav`) — never mock-only toys.
    """

    def __init__(self, workspace_root: Path | str = Path(r"c:\Elysia")) -> None:
        self.workspace_root = Path(workspace_root)

    def probe_real_python_subprocess(
        self,
        protocol: InteractionProtocol,
        workloads: Sequence[int] = (5000, 15000, 35000, 70000),
    ) -> Dict[str, Any]:
        """
        Invokes genuine external Python OS subprocesses across increasing workloads
        to discover lived compute-scale correlations + tests non-crashing error capture.
        """
        def _exec_external(stim: Dict[str, float]) -> Dict[str, float]:
            n_iter = int(round(stim["compute_iterations"]))
            code = (
                "import time, math\n"
                "t0 = time.perf_counter_ns()\n"
                f"acc = sum(math.sin(i * 0.001) for i in range({n_iter}))\n"
                "dt_ms = (time.perf_counter_ns() - t0) / 1e6\n"
                "print(f'{dt_ms:.6f},{acc:.6f}')\n"
            )
            t_wall0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            wall_ms = (time.perf_counter() - t_wall0) * 1000.0
            if proc.returncode != 0:
                raise RuntimeError(f"Subprocess exited {proc.returncode}: {proc.stderr.strip()}")
            parts = proc.stdout.strip().split(",")
            return {
                "inner_compute_ms": float(parts[0]),
                "external_checksum": float(parts[1]),
                "wall_ms": float(wall_ms),
            }

        ok_res = protocol.interact_and_discover_axis(
            medium_name="real_python_subprocess",
            interact_fn=_exec_external,
            stimulus_channel="compute_iterations",
            sweep_values=[float(w) for w in workloads],
            scale_tier="sky",
            scale_radius=3.5,
        )

        # Deliberate unexpected external process failure -> Guardrail §5 (zero crash!)
        def _failing_external(_stim: Dict[str, float]) -> Dict[str, float]:
            proc = subprocess.run(
                [sys.executable, "-c", "1/0"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"External process ZeroDivisionError (rc={proc.returncode})")
            return {"ok": 1.0}

        fail_res = protocol.interact_and_discover_axis(
            medium_name="real_subprocess_fault_probe",
            interact_fn=_failing_external,
            stimulus_channel="fault_probe",
            sweep_values=[1.0],
            scale_tier="sky",
            scale_radius=4.0,
        )
        return {
            "compute_axis": ok_res["axis"],
            "compute_spectrum": ok_res["correlation_spectrum"],
            "fault_boundary_events": fail_res["friction_events"],
            "summary_ko": (
                f"실제 OS 파이썬 서브프로세스({len(workloads)}회 실행) 연산 규모와 실행 지연 상관 축을 "
                f"스스로 형성했고, 의도적 외부 프로세스 오류({len(fail_res['friction_events'])}건)도 "
                "강제 종료 없이 경계 학습 사건으로 흡수했습니다."
            ),
        }

    def probe_real_workspace_files_and_media(
        self,
        protocol: InteractionProtocol,
    ) -> Dict[str, Any]:
        """Reads real `.py` ASTs and real `.jpg`/`.wav` files on disk into the open sphere."""
        topo_dir = self.workspace_root / "L1_causal_field" / "core" / "topology"
        py_files = sorted(p for p in topo_dir.glob("*.py") if p.name != "__init__.py")[:8]
        file_stats: List[Tuple[int, int, int, str]] = []
        for p in py_files:
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(src, filename=str(p))
                n_lines = len(src.splitlines())
                n_ast = sum(1 for _ in ast.walk(tree))
                n_defs = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
                file_stats.append((n_lines, n_ast, n_defs, p.stem))
            except Exception:
                continue

        file_stats.sort(key=lambda item: item[0])
        stat_map = {float(item[0]): item for item in file_stats}

        code_axis = None
        if len(stat_map) >= 3:
            def _probe_code_scale(stim: Dict[str, float]) -> Dict[str, float]:
                item = stat_map[float(stim["source_line_count"])]
                return {"ast_node_count": float(item[1]), "definition_count": float(item[2])}

            c_res = protocol.interact_and_discover_axis(
                medium_name="real_workspace_topology_code",
                interact_fn=_probe_code_scale,
                stimulus_channel="source_line_count",
                sweep_values=list(stat_map.keys()),
                scale_tier="water",
                scale_radius=2.6,
            )
            code_axis = c_res["axis"]

        media_droplets: List[CausalDroplet] = []
        face_path = self.workspace_root / "demos" / "real_world_assets" / "real_human_face_hd.jpg"
        if face_path.is_file():
            with Image.open(face_path) as im:
                rgb = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float64) / 255.0
            lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            hist, _ = np.histogram(lum, bins=8, range=(0.0, 1.0))
            sig = hist.astype(np.float64) + 1e-6
            d_img = CausalDroplet(
                droplet_id="external_media::real_human_face_hd",
                constituents=("rgb_luminance_field", "spatial_contrast_gradient"),
                internal_bonds=((0, 1, 0.88),),
                surface_tension=float(min(1.0, 0.4 + 2.0 * float(np.std(lum)))),
                scale_radius=2.8,
                spectral_signature=sig,
                local_order_notes={"file": str(face_path), "mean_lum": round(float(np.mean(lum)), 4)},
            )
            protocol.memory.absorb_droplet(d_img)
            media_droplets.append(d_img)

        wav_path = self.workspace_root / "data" / "unified_creation_dawn.wav"
        if wav_path.is_file():
            with wave.open(str(wav_path), "rb") as wf:
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                raw_b = wf.readframes(min(wf.getnframes(), 2048))
            if sw == 2:
                arr = np.frombuffer(raw_b, dtype=np.int16).astype(np.float64) / 32768.0
            else:
                arr = np.frombuffer(raw_b, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
            mono = arr[:: max(1, ch)][:1024]
            mag = np.abs(np.fft.rfft(mono))
            bands = np.array_split(mag, 8)
            sig_a = np.array([float(np.mean(b)) for b in bands], dtype=np.float64) + 1e-6
            d_wav = CausalDroplet(
                droplet_id="external_media::unified_creation_dawn",
                constituents=("acoustic_time_wave", "harmonic_spectrum_8band"),
                internal_bonds=((0, 1, 0.90),),
                surface_tension=float(min(1.0, 0.45 + float(np.std(mono)))),
                scale_radius=2.9,
                spectral_signature=sig_a,
                local_order_notes={"file": str(wav_path), "samples_read": len(mono)},
            )
            protocol.memory.absorb_droplet(d_wav)
            media_droplets.append(d_wav)

        return {
            "code_axis": code_axis,
            "files_scanned": len(file_stats),
            "media_droplets": media_droplets,
            "summary_ko": (
                f"실제 워크스페이스 소스 {len(file_stats)}개(라인-AST 상관 축 형성={code_axis is not None})와 "
                f"실물 매체 {len(media_droplets)}종(jpg·wav)을 구면 가상 관계 메모리에 흡수했습니다."
            ),
        }


# ---------------------------------------------------------------------------
# 6. Top-Level Unified ELYSIA v3.1 Orchestrator
# ---------------------------------------------------------------------------


class ElysiaV31OpenScaleRotorSystem:
    """Unifies ELYSIA v3.1 Open Scale-Rotor scenarios with LivingCycle & CausalField."""

    def __init__(
        self,
        workspace_root: Path | str = Path(r"c:\Elysia"),
        living_cycle: Optional[ElysiaUnifiedLivingCycle] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.memory = OpenVirtualRelationalMemory()
        self.protocol = InteractionProtocol(self.memory)
        self.external_bridge = RealExternalToolBridge(self.workspace_root)
        self.living_cycle = living_cycle

    def _seal_witness(self, tag: str) -> None:
        if self.living_cycle is None:
            return
        from core.orchestration.elysia_unified_living_cycle import LivingEncounter
        from core.orchestration.extended_cognitive_organism import ExtendedWorldState
        from core.physics.spectral_scale_rotor_manifold import encode_waveform_to_scale_rotors

        w = ExtendedWorldState(
            source_pos_x=self.living_cycle.organism.body_pos_x + 10.0,
            source_temp_k=3200.0,
        )
        self.living_cycle.breathe_and_understand(
            LivingEncounter(encounter_name=tag, world=w, active_organs=("eye", "ear", "hand"))
        )
        if self.living_cycle.memory_runtime is not None:
            t = np.linspace(0.0, 1.0, 64, endpoint=False)
            n = max(1, len(self.memory.nodes))
            sig = np.sin(2.0 * math.pi * (1.0 + 0.5 * n) * t) + 0.3 * np.cos(2.0 * math.pi * len(self.memory.weave_links) * t)
            man = encode_waveform_to_scale_rotors(sig, domain=(0.0, 1.0), max_macro_modes=8, max_levels=2, max_knots_per_level=4)
            self.living_cycle.memory_runtime.ingest_to_ice(f"v31_{tag}", man)

    def run_scenario_1_spectrum_and_fog(self) -> Dict[str, Any]:
        medium = OpaqueSensoryMedium(seed=11)
        taste_res = self.protocol.interact_and_discover_axis(
            medium_name="unlabeled_chemical_medium",
            interact_fn=medium.interact,
            stimulus_channel="substance_dose",
            sweep_values=[0.0, 0.5, 1.5, 3.0, 5.0, 2.0, 4.0],
            scale_tier="earth",
            scale_radius=1.1,
        )
        fog_env = FoggyRoadMultiFieldEnvironment(fog_density=0.92, seed=23)
        fog_res = self.protocol.assess_and_select_viable_modality(
            modalities={
                "visible_optical_field": fog_env.probe_optical,
                "em_radar_field": fog_env.probe_em_radar,
            },
            stimulus_channel="obstacle_distance_m",
            sweep_values=[5.0, 12.0, 20.0, 35.0, 50.0, 70.0],
        )
        self._seal_witness("v31_scenario1_spectrum_fog")
        return {"spectrum_discovery": taste_res, "fog_modality_assessment": fog_res}

    def run_scenario_2_apple_and_hierarchy(self) -> Dict[str, Any]:
        # 2A: Label-free spatial object principle ("사과 원리")
        apple_res = derive_structural_principle_label_free(
            self.protocol, UnlabeledSpatialObjectField()
        )

        # 2B: Atom-Molecule-Macro self-structure discernment & scale-invariant weaving
        a1 = CausalDroplet("unit_alpha_1", scale_radius=0.8, surface_tension=0.45, spectral_signature=np.array([1, 0.2, 0.1, 0, 0, 0, 0, 0.2]), open_valences=2)
        a2 = CausalDroplet("unit_alpha_2", scale_radius=0.85, surface_tension=0.48, spectral_signature=np.array([0.8, 0.4, 0.1, 0.1, 0, 0, 0, 0.1]), open_valences=2)
        a3 = CausalDroplet("unit_alpha_3", scale_radius=0.82, surface_tension=0.46, spectral_signature=np.array([0.9, 0.3, 0.2, 0, 0, 0, 0, 0.1]), open_valences=2)
        a4 = CausalDroplet("unit_alpha_4", scale_radius=0.88, surface_tension=0.47, spectral_signature=np.array([0.7, 0.5, 0.2, 0.1, 0, 0, 0, 0.1]), open_valences=2)
        a5 = CausalDroplet("unit_alpha_5", scale_radius=0.90, surface_tension=0.49, spectral_signature=np.array([0.6, 0.4, 0.3, 0.2, 0, 0, 0, 0.1]), open_valences=2)

        m1 = a1.bond_with(a2, "composite_m1")
        m2 = a3.bond_with(a4, "composite_m2")
        macro_pair = m1.bond_with(m2, "macro_cluster_0")
        g1 = macro_pair.bond_with(a5, "macro_enclosure_g1")

        for dr in (a1, a2, a3, a4, a5, m1, m2, macro_pair, g1):
            self.memory.absorb_droplet(dr)

        discernments = {
            "m1_vs_a3": m1.discern_against_self(a3),
            "m1_vs_m2": m1.discern_against_self(m2),
            "m1_vs_g1": m1.discern_against_self(g1),
        }
        self._seal_witness("v31_scenario2_apple_hierarchy")
        return {
            "apple_principle": apple_res,
            "discernments": discernments,
            "droplets": {"a3": a3, "m1": m1, "m2": m2, "g1": g1},
        }

    def run_scenario_3_real_external_world(self) -> Dict[str, Any]:
        before_metrics = self.memory.topology_metrics()
        subproc_res = self.external_bridge.probe_real_python_subprocess(self.protocol)
        ws_res = self.external_bridge.probe_real_workspace_files_and_media(self.protocol)
        after_metrics = self.memory.topology_metrics()
        self._seal_witness("v31_scenario3_external_world")
        return {
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "subprocess_probe": subproc_res,
            "workspace_probe": ws_res,
        }

    def run_all_v31_scenarios(self) -> Dict[str, Any]:
        s1 = self.run_scenario_1_spectrum_and_fog()
        s2 = self.run_scenario_2_apple_and_hierarchy()
        s3 = self.run_scenario_3_real_external_world()
        return {
            "scenario_1": s1,
            "scenario_2": s2,
            "scenario_3": s3,
            "final_topology": self.memory.topology_metrics(),
        }
