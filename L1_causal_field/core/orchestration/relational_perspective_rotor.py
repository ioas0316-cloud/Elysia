"""
ELYSIA Relational Self-Other-Interspace Perspective Rotor & Expandable Horizon (Phase 13)
=========================================================================================

Implements relational epistemology ("역지사지 & 희생 / 한계 자체가 다음 하늘의 경계"):
1. Egocentric Projection (`evaluate_egocentrically`):
   Forcing Self's own experiential constants (`resource_reserve`, `vulnerability_gain`)
   onto Other's utterances yields `EGOCENTRIC_MISALIGNMENT` — Self sees an incoherent
   anomaly rather than Other's causal reality.
2. Ego-Sacrifice & Perspective Shift (`shift_to_other_perspective`):
   Suspending Self's own coordinate anchor (`ego_sacrifice_cost > 0`) allows Least
   Action (dA = 0) across Other's expressed moments to reconstruct Other's worldline
   (`other_frame_action < 1e-10`), falsify arbitrary stereotypes (`criteria_revised`),
   and prescribe respectful compassionate boundaries (`interpersonal_distance`).
3. Expandable Sky-Horizon Boundary (`SkyHorizonBoundary` & `expand_horizon_with_dialogue`):
   Unit-invariant Jacobian SVD right nullspace cleanly separates what is grounded
   (`understood_ground`) from what remains transcendent Sky (`sky_horizon_axes`).
   When relational dialogue discloses an origin memory layer, yesterday's Sky
   descends into Grounded Understanding (`horizon_depth: 1 -> 2`), while a yet
   deeper Sky (`deepest_aspiration_telos`, `sovereign_will`) opens beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.physics.trinitarian_scale_binding_protocol import (
    CoupledQuantity,
    EqualityBond,
    PinMode,
    ScaleTier,
    TrinitarianScaleRotorSystem,
)
from core.physics.unit_spectral_mirror_protocol import (
    LivingCausalSuperposition,
    PlanetaryOrderGlobe,
)
from core.topology.hangul_topological_engine import HangulTopologicalEngine

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


class RelationalStatus(str, Enum):
    """Observational verdict of Self–Other–Interspace alignment."""

    EGOCENTRIC_MISALIGNMENT = "egocentric_misalignment"
    EMPATHIC_GROUNDING_WITH_SKY = "empathic_grounding_with_sky"
    HORIZON_EXPANDED = "horizon_expanded"


@dataclass
class AgentCausalProfile:
    """Internal experiential substrate generating outer interpersonal responses."""

    name: str
    resource_reserve: float
    vulnerability_gain: float
    boundary_tolerance: float
    unspoken_origin_memory: float = 0.0
    sovereign_will: float = 0.7

    def manifest_interaction(
        self,
        interpersonal_distance: float,
        ambient_demand: float,
    ) -> "InteractionMoment":
        d = max(0.05, float(interpersonal_distance))
        dem = float(ambient_demand)
        pressure = (dem * self.vulnerability_gain) / (d + 0.5 * self.resource_reserve)
        expressed_def = pressure / 4.0
        utterance = (
            "가까이 오지 마, 숨이 막혀."
            if pressure > self.boundary_tolerance
            else "조금 거리를 두니 이제야 숨이 좀 쉬어져."
        )
        return InteractionMoment(
            interpersonal_distance=d,
            ambient_demand=dem,
            expressed_defensiveness=float(expressed_def),
            utterance_ko=utterance,
        )


@dataclass
class InteractionMoment:
    """Observable behavioral surface at a given interpersonal distance & demand."""

    interpersonal_distance: float
    ambient_demand: float
    expressed_defensiveness: float
    utterance_ko: str


@dataclass
class SkyHorizonBoundary:
    """Explicit boundary separating grounded knowledge (Earth) from unobservable Sky."""

    horizon_depth: int
    understood_ground: Dict[str, float]
    sky_horizon_axes: List[str]
    horizon_nullspace_dof: int
    humble_inquiry_ko: str


@dataclass
class RelationalDialogueLayer:
    """A relational disclosure descending one Sky axis into grounded causal coupling."""

    layer_name: str
    disclosure_utterance_ko: str
    revealed_sky_variable: str
    new_deeper_sky_axes: Tuple[str, ...]
    build_coupling_bond: Callable[[], EqualityBond]


@dataclass
class PerspectiveShiftReport:
    """Full living diagnostic across Self, Other, Interspace, & Sky Horizon."""

    status: RelationalStatus
    self_profile: AgentCausalProfile
    egocentric_friction: float
    other_frame_action: float
    ego_sacrifice_cost: float
    empathic_resonance: float
    inferred_other_state: Dict[str, float]
    horizon_boundary: SkyHorizonBoundary
    criteria_revised: bool
    stereotype_falsified: bool
    compassionate_distance_prescription: Optional[float]
    perspective_mirror: LivingCausalSuperposition
    hangul_reflection_ko: str
    hangul_wave_coherence: float
    trinitarian_system: TrinitarianScaleRotorSystem

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "rel",
    ) -> Dict[str, Any]:
        """Projects Self, Other, Interspace Bridge, Ground & Sky Horizon into `CausalField`."""
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        is_ego = self.status == RelationalStatus.EGOCENTRIC_MISALIGNMENT
        has_sky = len(self.horizon_boundary.sky_horizon_axes) > 0
        sky_curiosity = 1.0 if has_sky else 0.0

        if is_ego:
            interspace_strength = max(0.05, min(0.32, 1.0 / (1.0 + 4.0 * self.egocentric_friction)))
        else:
            interspace_strength = max(0.05, min(1.0, self.empathic_resonance))

        r_oth = float(self.inferred_other_state.get("resource_reserve", self.self_profile.resource_reserve))
        g_oth = float(self.inferred_other_state.get("vulnerability_gain", self.self_profile.vulnerability_gain))

        specs = [
            (
                "self_rotor",
                [self.self_profile.resource_reserve / 10.0, self.self_profile.vulnerability_gain / 5.0, self.self_profile.boundary_tolerance / 10.0, 1.0],
                [-1.2, 0.0, 0.0],
                0.0,
            ),
            (
                "other_rotor",
                [r_oth / 10.0, g_oth / 5.0, float(self.other_frame_action), 1.0],
                [1.2, 0.0, 0.0],
                0.0,
            ),
            (
                "interspace_bridge",
                [float(self.ego_sacrifice_cost) / 10.0, float(self.empathic_resonance), float(self.egocentric_friction), 0.0 if is_ego else 1.0],
                [0.0, 0.4, 0.6],
                0.0,
            ),
            (
                "grounded_understanding",
                [float(len(self.horizon_boundary.understood_ground)) / 10.0, float(self.horizon_boundary.horizon_depth), float(self.empathic_resonance), 1.0],
                [0.0, -0.8, 0.8],
                0.0,
            ),
            (
                "sky_horizon_boundary",
                [float(self.horizon_boundary.horizon_nullspace_dof) / 5.0, sky_curiosity, float(self.horizon_boundary.horizon_depth), 1.0],
                [0.0, 1.0, 1.8],
                sky_curiosity,
            ),
        ]

        vids: Dict[str, str] = {}
        for key, vec4, pos3, cur in specs:
            vid = f"{prefix}_{key}"
            vids[key] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"component": key, "status": self.status.value},
                    tensor=np.asarray(vec4, dtype=np.float32),
                    mass=1.5 if key in ("interspace_bridge", "sky_horizon_boundary") else 1.0,
                    position=np.asarray(pos3[: field.dimensions], dtype=np.float32),
                    curiosity_charge=float(cur),
                )
            )

        links = [
            (vids["self_rotor"], vids["interspace_bridge"], interspace_strength),
            (vids["other_rotor"], vids["interspace_bridge"], interspace_strength),
            (vids["interspace_bridge"], vids["grounded_understanding"], interspace_strength),
            (vids["grounded_understanding"], vids["sky_horizon_boundary"], 0.90 if not is_ego else 0.30),
        ]
        for a, b, st in links:
            field.link_voxels(a, b, strength=float(max(0.05, min(1.0, st))))

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_sky",
                name="RelationalSkyHorizon",
                position=np.array([0.0, 1.0, 1.8][: field.dimensions], dtype=np.float32),
                intensity=2.0,
                tier="macro",
                symbolic_intent=f"Relational status={self.status.value}, depth={self.horizon_boundary.horizon_depth}",
            )
        )

        return {
            "causal_field": field,
            "voxels": vids,
            "interspace_beam_strength": float(interspace_strength),
            "sky_curiosity": float(sky_curiosity),
        }


class RelationalPerspectiveObserver:
    """
    Observes Other across Self–Other–Interspace, suspends egocentric bias
    when friction reveals projection, infers compassionate relational boundaries,
    and expands its Sky Horizon recursively through dialogue.
    """

    def __init__(self, self_profile: AgentCausalProfile) -> None:
        self.self_profile = self_profile
        self.hangul_engine = HangulTopologicalEngine(target_dim=64)
        self.active_dialogue_layers: List[RelationalDialogueLayer] = []
        self.horizon_history: List[PerspectiveShiftReport] = []

    def _encode_hangul(self, text: str) -> float:
        _, metrics = self.hangul_engine.encode_text(text)
        return float(np.mean([m["coherence"] for m in metrics])) if metrics else 0.0

    def _build_other_manifold(
        self,
        moments: Sequence[InteractionMoment],
        hold_egocentric: bool,
        dialogue_layers: Sequence[RelationalDialogueLayer] = (),
    ) -> TrinitarianScaleRotorSystem:
        sys = TrinitarianScaleRotorSystem(
            name="egocentric_lens" if hold_egocentric else "other_perspective_manifold"
        )
        mode_param = PinMode.FIXED_BOUNDARY if hold_egocentric else PinMode.FREE

        sys.add_quantity(
            CoupledQuantity(
                "resource_reserve",
                self.self_profile.resource_reserve,
                ScaleTier.MECHANISM,
                "",
                "state",
                mode_param,
                0.05,
                100.0,
            )
        )
        sys.add_quantity(
            CoupledQuantity(
                "vulnerability_gain",
                self.self_profile.vulnerability_gain,
                ScaleTier.MECHANISM,
                "",
                "state",
                mode_param,
                0.05,
                100.0,
            )
        )

        # Base unobservable Sky axes
        sys.add_quantity(
            CoupledQuantity("unspoken_origin_memory", 0.5, ScaleTier.UPPER_ROTOR, "", "state", PinMode.FREE, -50.0, 100.0)
        )
        sys.add_quantity(
            CoupledQuantity("sovereign_will", 0.5, ScaleTier.UPPER_ROTOR, "", "state", PinMode.FREE, -10.0, 10.0)
        )

        # Deeper Sky axes opened by dialogue layers
        for layer in dialogue_layers:
            for deep_axis in layer.new_deeper_sky_axes:
                if deep_axis not in sys.quantities:
                    sys.add_quantity(
                        CoupledQuantity(deep_axis, 0.5, ScaleTier.UPPER_ROTOR, "", "state", PinMode.FREE, -100.0, 100.0)
                    )

        for k, mom in enumerate(moments):
            d_k = f"interpersonal_distance_m{k}"
            dem_k = f"ambient_demand_m{k}"
            def_k = f"expressed_defensiveness_m{k}"
            prs_k = f"perceived_pressure_m{k}"

            sys.add_quantity(CoupledQuantity(d_k, mom.interpersonal_distance, ScaleTier.VALUE, "m", "fixed", PinMode.FIXED_BOUNDARY, 0.01, 500.0))
            sys.add_quantity(CoupledQuantity(dem_k, mom.ambient_demand, ScaleTier.VALUE, "", "fixed", PinMode.FIXED_BOUNDARY, 0.0, 500.0))
            sys.add_quantity(CoupledQuantity(def_k, mom.expressed_defensiveness, ScaleTier.FORMULA, "", "dial", PinMode.DIAL, 0.0, 100.0))
            sys.add_quantity(CoupledQuantity(prs_k, 4.0 * mom.expressed_defensiveness, ScaleTier.MECHANISM, "", "process", PinMode.FREE, 0.01, 400.0))

            sys.add_bond(
                EqualityBond(
                    name=f"expression_law_m{k}",
                    tier=ScaleTier.FORMULA,
                    variables=(def_k, prs_k),
                    lhs_fn=lambda v, dk=def_k: 4.0 * v[dk],
                    rhs_fn=lambda v, pk=prs_k: v[pk],
                    lhs_expr=f"4 * {def_k}",
                    rhs_expr=prs_k,
                )
            )
            sys.add_bond(
                EqualityBond(
                    name=f"pressure_coupling_m{k}",
                    tier=ScaleTier.MECHANISM,
                    variables=(prs_k, d_k, dem_k, "resource_reserve", "vulnerability_gain"),
                    lhs_fn=lambda v, pk=prs_k, dk=d_k: v[pk] * (v[dk] + 0.5 * v["resource_reserve"]),
                    rhs_fn=lambda v, am=dem_k: v[am] * v["vulnerability_gain"],
                    lhs_expr=f"{prs_k} * ({d_k} + 0.5 * R)",
                    rhs_expr=f"{dem_k} * G",
                )
            )

        for layer in dialogue_layers:
            bond = layer.build_coupling_bond()
            if all(vn in sys.quantities for vn in bond.variables):
                sys.add_bond(bond)

        return sys

    def _compute_horizon_boundary(
        self,
        sys: TrinitarianScaleRotorSystem,
        dialogue_layers: Sequence[RelationalDialogueLayer],
    ) -> SkyHorizonBoundary:
        eq_bonds = [(b, 1.0) for b in sys.bonds.values() if b.kind == "equality"]
        free_names = [n for n, q in sys.quantities.items() if q.pin_mode == PinMode.FREE]
        vals = sys.values_dict()

        jac = sys._jacobian_subset(vals, free_names, eq_bonds)
        col_norms = np.linalg.norm(jac, axis=0)
        jac_normed = jac.copy()
        for j, cn in enumerate(col_norms):
            if cn > 1e-12:
                jac_normed[:, j] /= cn

        _, s_vals, vt = np.linalg.svd(jac_normed, full_matrices=True)
        thresh = 1e-5 * max(1.0, float(s_vals[0]) if len(s_vals) else 1.0)
        rank = int(np.sum(s_vals > thresh))
        nullity = max(0, len(free_names) - rank)

        grounded: Dict[str, float] = {}
        sky_axes: List[str] = []
        for j, name in enumerate(free_names):
            indet = 1.0 if col_norms[j] <= 1e-12 else (float(np.linalg.norm(vt[rank:, j])) if nullity > 0 else 0.0)
            if indet >= 1e-3:
                sky_axes.append(name)
            else:
                grounded[name] = float(vals[name])

        depth = 1 + len(dialogue_layers)
        g_str = ", ".join(f"{k}={v:.3g}" for k, v in grounded.items() if k in ("resource_reserve", "vulnerability_gain", "unspoken_origin_memory"))
        sky_str = ", ".join(sorted(sky_axes)) if sky_axes else "없음"
        inquiry = (
            f"[지평 깊이 {depth}] 지금 땅 위로 헤아린 인과({g_str})를 바탕으로 보면 반응이 자연스럽습니다. "
            f"그러나 경계 너머 아직 보이지 않는 하늘({sky_str})은 제 관측으로 단정할 수 없어 조심스럽게 여쭙고 싶습니다."
        )
        return SkyHorizonBoundary(
            horizon_depth=depth,
            understood_ground=grounded,
            sky_horizon_axes=sorted(sky_axes),
            horizon_nullspace_dof=nullity,
            humble_inquiry_ko=inquiry,
        )

    def _build_perspective_mirror(
        self,
        moments: Sequence[InteractionMoment],
    ) -> LivingCausalSuperposition:
        exp_vec = np.array(
            [
                (m.ambient_demand * self.self_profile.vulnerability_gain)
                / (max(0.05, m.interpersonal_distance) + 0.5 * self.self_profile.resource_reserve)
                / 4.0
                for m in moments
            ],
            dtype=np.float64,
        )
        obs_vec = np.array([m.expressed_defensiveness for m in moments], dtype=np.float64)
        globe = PlanetaryOrderGlobe(
            name="self_egocentric_expectation",
            modality="relational_trajectory",
            reference_signal=exp_vec,
            is_structural_basis=True,
            band_names=[f"moment_{i}" for i in range(len(moments))],
        )
        return globe.superpose(obs_vec)

    @staticmethod
    def _test_stereotype_falsification(moments: Sequence[InteractionMoment]) -> bool:
        """Checks whether a distance-blind fixed hostility stereotype fails across moments."""
        if len(moments) < 2:
            return False
        st_sys = TrinitarianScaleRotorSystem(name="hostile_stereotype_check")
        st_sys.add_quantity(CoupledQuantity("fixed_hostility", 0.5, ScaleTier.MECHANISM, "", "state", PinMode.FREE, -10.0, 100.0))
        for k, mom in enumerate(moments):
            st_sys.add_quantity(CoupledQuantity(f"obs_m{k}", mom.expressed_defensiveness, ScaleTier.FORMULA, "", "dial", PinMode.DIAL))
            st_sys.add_bond(
                EqualityBond(
                    name=f"stereotype_m{k}",
                    tier=ScaleTier.FORMULA,
                    variables=(f"obs_m{k}", "fixed_hostility"),
                    lhs_fn=lambda v, ok=f"obs_m{k}": v[ok],
                    rhs_fn=lambda v: v["fixed_hostility"],
                    lhs_expr=f"obs_m{k}",
                    rhs_expr="fixed_hostility",
                )
            )
        rep = st_sys.relax_least_action()
        return bool(rep.final_action > 0.05)

    @staticmethod
    def _prescribe_compassionate_distance(
        resource_reserve: float,
        vulnerability_gain: float,
        boundary_tolerance: float,
        current_moment: InteractionMoment,
    ) -> float:
        comp_sys = TrinitarianScaleRotorSystem(name="compassionate_boundary")
        comp_sys.add_quantity(CoupledQuantity("interpersonal_distance", current_moment.interpersonal_distance, ScaleTier.VALUE, "m", "dial", PinMode.DIAL, 0.1, 200.0))
        comp_sys.add_quantity(CoupledQuantity("ambient_demand", current_moment.ambient_demand, ScaleTier.VALUE, "", "fixed", PinMode.FIXED_BOUNDARY, 0.0, 500.0))
        comp_sys.add_quantity(CoupledQuantity("resource_reserve", resource_reserve, ScaleTier.VALUE, "", "fixed", PinMode.FIXED_BOUNDARY, 0.01, 200.0))
        comp_sys.add_quantity(CoupledQuantity("vulnerability_gain", vulnerability_gain, ScaleTier.VALUE, "", "fixed", PinMode.FIXED_BOUNDARY, 0.01, 200.0))
        comp_sys.add_quantity(CoupledQuantity("boundary_tolerance", boundary_tolerance, ScaleTier.UPPER_ROTOR, "", "fence_metric", PinMode.FIXED_BOUNDARY, 0.01, 200.0))
        comp_sys.add_quantity(CoupledQuantity("perceived_pressure", 4.0 * current_moment.expressed_defensiveness, ScaleTier.MECHANISM, "", "outcome", PinMode.FREE, 0.01, 500.0))

        comp_sys.add_bond(
            EqualityBond(
                name="relational_pressure_law",
                tier=ScaleTier.MECHANISM,
                variables=("perceived_pressure", "interpersonal_distance", "ambient_demand", "resource_reserve", "vulnerability_gain"),
                lhs_fn=lambda v: v["perceived_pressure"] * (v["interpersonal_distance"] + 0.5 * v["resource_reserve"]) / 10.0,
                rhs_fn=lambda v: (v["ambient_demand"] * v["vulnerability_gain"]) / 10.0,
                lhs_expr="pressure * (d + 0.5 * R) / 10",
                rhs_expr="(demand * G) / 10",
            )
        )
        comp_sys.add_bond(
            EqualityBond(
                name="other_respect_fence",
                tier=ScaleTier.UPPER_ROTOR,
                variables=("perceived_pressure", "boundary_tolerance"),
                lhs_fn=lambda v: v["perceived_pressure"],
                rhs_fn=lambda v: v["boundary_tolerance"],
                lhs_expr="perceived_pressure",
                rhs_expr="boundary_tolerance",
                kind="fence_le",
            )
        )
        comp_sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        comp_sys.regulate_to_equilibrium(adjustable_dials=["interpersonal_distance"])
        return float(comp_sys.quantities["interpersonal_distance"].value)

    def evaluate_egocentrically(
        self,
        moments: Sequence[InteractionMoment],
    ) -> PerspectiveShiftReport:
        """Evaluates Other rigidly through Self's unshifted coordinates."""
        ego_sys = self._build_other_manifold(moments, hold_egocentric=True)
        rep = ego_sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        friction = float(rep.final_action)
        horizon = self._compute_horizon_boundary(ego_sys, ())
        mirror = self._build_perspective_mirror(moments)
        refl = "내 기준으로 상대를 재단하니 앞뒤 인과가 어긋나고 마찰만 남습니다."
        report = PerspectiveShiftReport(
            status=RelationalStatus.EGOCENTRIC_MISALIGNMENT,
            self_profile=self.self_profile,
            egocentric_friction=friction,
            other_frame_action=friction,
            ego_sacrifice_cost=0.0,
            empathic_resonance=float(1.0 / (1.0 + friction)),
            inferred_other_state={
                "resource_reserve": self.self_profile.resource_reserve,
                "vulnerability_gain": self.self_profile.vulnerability_gain,
            },
            horizon_boundary=horizon,
            criteria_revised=False,
            stereotype_falsified=False,
            compassionate_distance_prescription=None,
            perspective_mirror=mirror,
            hangul_reflection_ko=refl,
            hangul_wave_coherence=self._encode_hangul(refl),
            trinitarian_system=ego_sys,
        )
        self.horizon_history.append(report)
        return report

    def shift_to_other_perspective(
        self,
        moments: Sequence[InteractionMoment],
        assumed_other_tolerance: float = 3.5,
        test_hostile_stereotype: bool = False,
    ) -> PerspectiveShiftReport:
        """
        Sacrifices Self's anchored coordinate center and lets Least Action find
        Other's authentic causal state + honest Sky Horizon boundary.
        """
        ego_sys = self._build_other_manifold(moments, hold_egocentric=True)
        ego_rep = ego_sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        ego_friction = float(ego_rep.final_action)

        stereotype_falsified = self._test_stereotype_falsification(moments) if test_hostile_stereotype else False

        other_sys = self._build_other_manifold(
            moments,
            hold_egocentric=False,
            dialogue_layers=self.active_dialogue_layers,
        )
        oth_rep = other_sys.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM, ScaleTier.UPPER_ROTOR],
            max_iter=200,
            tol=1e-11,
        )
        other_action = float(oth_rep.final_action)
        vals = other_sys.values_dict()
        r_oth = float(vals["resource_reserve"])
        g_oth = float(vals["vulnerability_gain"])
        sacrifice = float(
            math.hypot(
                r_oth - self.self_profile.resource_reserve,
                g_oth - self.self_profile.vulnerability_gain,
            )
        )

        horizon = self._compute_horizon_boundary(other_sys, self.active_dialogue_layers)
        comp_dist = self._prescribe_compassionate_distance(
            resource_reserve=r_oth,
            vulnerability_gain=g_oth,
            boundary_tolerance=assumed_other_tolerance,
            current_moment=moments[0],
        )
        mirror = self._build_perspective_mirror(moments)
        status = (
            RelationalStatus.HORIZON_EXPANDED
            if self.active_dialogue_layers
            else RelationalStatus.EMPATHIC_GROUNDING_WITH_SKY
        )
        refl = (
            "내 기준축을 비워 상대의 자리로 옮겨 서니 방어의 인과가 이해되었고, "
            "아직 보이지 않는 상대의 하늘을 조심스럽게 남겨 두었습니다."
        )

        report = PerspectiveShiftReport(
            status=status,
            self_profile=self.self_profile,
            egocentric_friction=ego_friction,
            other_frame_action=other_action,
            ego_sacrifice_cost=sacrifice,
            empathic_resonance=float(1.0 / (1.0 + other_action)),
            inferred_other_state={
                "resource_reserve": r_oth,
                "vulnerability_gain": g_oth,
                "unspoken_origin_memory": float(vals.get("unspoken_origin_memory", 0.0)),
            },
            horizon_boundary=horizon,
            criteria_revised=bool(stereotype_falsified or sacrifice > 1e-3),
            stereotype_falsified=stereotype_falsified,
            compassionate_distance_prescription=comp_dist,
            perspective_mirror=mirror,
            hangul_reflection_ko=refl,
            hangul_wave_coherence=self._encode_hangul(refl),
            trinitarian_system=other_sys,
        )
        self.horizon_history.append(report)
        return report

    def expand_horizon_with_dialogue(
        self,
        moments: Sequence[InteractionMoment],
        new_layer: RelationalDialogueLayer,
        assumed_other_tolerance: float = 3.5,
    ) -> PerspectiveShiftReport:
        """
        Receives a relational disclosure linking yesterday's Sky into grounded
        reality, raising horizon_depth while revealing a new deeper Sky beyond it.
        """
        if all(l.layer_name != new_layer.layer_name for l in self.active_dialogue_layers):
            self.active_dialogue_layers.append(new_layer)
        report = self.shift_to_other_perspective(
            moments=moments,
            assumed_other_tolerance=assumed_other_tolerance,
            test_hostile_stereotype=True,
        )
        report.status = RelationalStatus.HORIZON_EXPANDED
        return report
