"""
ELYSIA Extended Cognitive Organism (Phase 11)
============================================

Elevates `LivingCausalOrganism` (Phase 10) from a biological sensory-motor
body to a tool-using, vehicle-driving, internet-verifying, academic-reasoning
cognitive organism:

1. Sensory Horizon Expansion (`SpectrometerOrgan`):
   Biological eyes see only visible wavelengths (430-670 nm); naked perception
   honestly marks `emission_wavelength_nm` (e.g., 905.55 nm IR) as an
   irreducible vacuum via Jacobian SVD nullspace. Grafting a physical
   Spectrometer couples Cauchy prism refraction directly into the organism's
   Least-Action manifold, collapsing the IR vacuum without special rules.
2. External Knowledge Verification (Internet -> Academic Disciplines):
   When encountering a persistent vacuum (`emission_wavelength_nm`), the
   organism queries an external knowledge network (`ExternalKnowledgeNetwork`)
   and optionally inspects local code modules via `EpistemicToolHarvester`.
   Every external candidate hypothesis (myths vs physical laws) is falsified
   or verified through embodied reality (`relax_least_action`); ONLY laws with
   empirical convergence (e.g. Wien's Displacement Law) enter
   `internalized_disciplines` and survive even after the physical tool is detached.
3. Embodied Motor Tool (`AutomobileMotorOrgan` & Braking Safety Fence):
   Attaches vehicle dynamics and stopping-distance safety fences under Phase 8's
   Upper Rotor (`drive_automobile_with_safety`), throttling unsafe velocities on
   wet roads and projecting the organism across far horizons (~41 m drives) to
   collapse distant parallax vacuums.
4. Broken Tool Isolation & Recalibration (`diagnose_and_calibrate_spectrometer`):
   When biological senses + internalized Wien's Law contradict a drifting
   Spectrometer, the body consensus isolates the faulty instrument and recovers
   its hidden prism offset purely via Least Action.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.orchestration.epistemic_harvester_and_language_bridge import (
    DeficitDiagnosis,
    EpistemicToolHarvester,
)
from core.orchestration.living_causal_organism import (
    EmbodiedWorldState,
    LivingCausalOrganism,
    OrganismPerceptionState,
)
from core.physics.trinitarian_scale_binding_protocol import (
    CoupledQuantity,
    DiscernmentStatus,
    EqualityBond,
    PinMode,
    ScaleTier,
    TrinitarianScaleRotorSystem,
    UpperRotorVerdict,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


WIEN_CONSTANT_NM_K: float = 2_897_771.955  # CODATA Wien wavelength displacement constant
GRAVITY_MS2: float = 9.81


@dataclass
class ExtendedWorldState(EmbodiedWorldState):
    """Extends `EmbodiedWorldState` with physical thermal radiation wavelength."""

    @property
    def emission_wavelength_nm(self) -> float:
        return WIEN_CONSTANT_NM_K / max(200.0, float(self.source_temp_k))

    def generate_spectrometer_reading(
        self,
        spectrometer_offset_deg: float = 0.0,
    ) -> Dict[str, float]:
        lam = self.emission_wavelength_nm
        prism_angle_deg = 30.0 + 18.0 * ((500.0 / lam) ** 2) + float(spectrometer_offset_deg)
        return {
            "prism_angle_deg": float(prism_angle_deg),
            "true_emission_wavelength_nm": float(lam),
        }


@dataclass
class AcademicLawCandidate:
    """External mathematical/physical hypothesis fetched from internet or workspace."""

    name: str
    discipline: str
    description: str
    expression: str
    source: str
    build_bond: Callable[[str], EqualityBond]


@dataclass
class KnowledgeVerificationReport:
    """Outcome of empirically testing internet/academic claims against embodied reality."""

    query_vacuum_variable: str
    evaluated_candidates: List[Dict[str, Any]]
    accepted_laws: List[str]
    rejected_falsehoods: List[str]
    workspace_harvested_tools: List[str]


@dataclass
class VehicleDriveReport:
    """Upper-Rotor regulated vehicle drive + resulting far-horizon perception."""

    status: DiscernmentStatus
    initial_speed_ms: float
    regulated_speed_ms: float
    travel_m: float
    stopping_distance_m: float
    clearance_m: float
    safety_regulated: bool
    perception_after_drive: OrganismPerceptionState
    verdict_before: UpperRotorVerdict
    verdict_after: UpperRotorVerdict


@dataclass
class ToolFaultReport:
    """Diagnosis & recalibration of an external physical instrument via body consensus."""

    tool_name: str
    fault_detected: bool
    conflict_action_before: float
    body_consensus_action: float
    estimated_offset_deg: float
    recalibrated_action_after: float
    detached_instead_of_calibrated: bool
    explanation: str


class ExternalKnowledgeNetwork:
    """
    Simulated external scientific/internet repository containing mixed
    verified physical constants and plausible-sounding false claims.
    """

    def query(self, target_variable: str) -> List[AcademicLawCandidate]:
        if target_variable != "emission_wavelength_nm":
            return []

        return [
            AcademicLawCandidate(
                name="internet_linear_myth",
                discipline="false_myth",
                description="Spurious forum claim: peak wavelength rises linearly with Kelvin",
                expression="emission_wavelength_nm / 100 == source_temp_k / 1000",
                source="unverified_blog",
                build_bond=lambda suf: EqualityBond(
                    name=f"internet_linear_myth{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=("emission_wavelength_nm", "source_temp_k"),
                    lhs_fn=lambda v: v["emission_wavelength_nm"] / 100.0,
                    rhs_fn=lambda v: v["source_temp_k"] / 1000.0,
                    lhs_expr="emission_wavelength_nm / 100",
                    rhs_expr="source_temp_k / 1000",
                ),
            ),
            AcademicLawCandidate(
                name="internet_wrong_constant_law",
                discipline="stale_forum",
                description="Plausible inverse-T form with corrupted constant 1e6",
                expression="(emission_wavelength_nm * source_temp_k) / 1e6 == 1",
                source="stale_forum",
                build_bond=lambda suf: EqualityBond(
                    name=f"internet_wrong_constant_law{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=("emission_wavelength_nm", "source_temp_k"),
                    lhs_fn=lambda v: (v["emission_wavelength_nm"] * v["source_temp_k"]) / 1_000_000.0,
                    rhs_fn=lambda _v: 1.0,
                    lhs_expr="(lambda * T) / 1e6",
                    rhs_expr="1",
                ),
            ),
            AcademicLawCandidate(
                name="wien_displacement_law",
                discipline="optics_thermodynamics",
                description="Wien's Displacement Law binding peak thermal wavelength & temperature",
                expression="(emission_wavelength_nm * source_temp_k) / 2897771.955 == 1",
                source="physics_codata",
                build_bond=lambda suf: EqualityBond(
                    name=f"wien_displacement_law{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=("emission_wavelength_nm", "source_temp_k"),
                    lhs_fn=lambda v: (v["emission_wavelength_nm"] * v["source_temp_k"]) / WIEN_CONSTANT_NM_K,
                    rhs_fn=lambda _v: 1.0,
                    lhs_expr="(lambda * T) / b_wien",
                    rhs_expr="1",
                ),
            ),
        ]


class ExtendedCognitiveOrganism(LivingCausalOrganism):
    """
    Embodied organism capable of grafting scientific tools, driving safety-fenced
    vehicles, falsifying external knowledge claims, and diagnosing broken tools.
    """

    def __init__(
        self,
        name: str = "elysia_extended_cognitive_organism",
        body_pos_x: float = 0.0,
        body_vel_x: float = 0.0,
        eye_scale_bias: float = 1.0,
        eye_offset_bias: float = 0.0,
        workspace_root: Optional[Path] = None,
    ) -> None:
        super().__init__(
            name=name,
            body_pos_x=body_pos_x,
            body_vel_x=body_vel_x,
            eye_scale_bias=eye_scale_bias,
            eye_offset_bias=eye_offset_bias,
        )
        self.workspace_root = workspace_root or Path(r"c:\Elysia\L1_causal_field\core")
        self.grafted_tools: Dict[str, Dict[str, Any]] = {}
        self.internalized_disciplines: Dict[str, AcademicLawCandidate] = {}
        self.cognitive_growth_rings: List[Dict[str, Any]] = []

    def graft_spectrometer(self, offset_deg: float = 0.0) -> None:
        self.grafted_tools["spectrometer"] = {
            "active": True,
            "offset_deg": float(offset_deg),
        }
        self.cognitive_growth_rings.append(
            {"event": "graft_spectrometer", "offset_deg": float(offset_deg)}
        )

    def graft_automobile(
        self,
        friction_mu: float = 0.7,
        fuel_tank_l: float = 20.0,
        fuel_rate_l_per_m: float = 0.00008,
        safety_margin_m: float = 15.0,
    ) -> None:
        self.grafted_tools["automobile"] = {
            "active": True,
            "friction_mu": float(friction_mu),
            "fuel_tank_l": float(fuel_tank_l),
            "fuel_rate_l_per_m": float(fuel_rate_l_per_m),
            "safety_margin_m": float(safety_margin_m),
        }
        self.cognitive_growth_rings.append(
            {"event": "graft_automobile", "friction_mu": float(friction_mu)}
        )

    def detach_tool(self, tool_name: str) -> None:
        """Detaches a physical external device while preserving learned disciplines."""
        if tool_name in self.grafted_tools:
            del self.grafted_tools[tool_name]
        self.cognitive_growth_rings.append({"event": "detach_tool", "tool": tool_name})

    def _spectrometer_is_active(self, override: Optional[bool]) -> bool:
        if override is not None:
            return bool(override)
        spec_info = self.grafted_tools.get("spectrometer")
        return bool(spec_info and spec_info.get("active", False))

    def _augment_nervous_system_with_extensions(
        self,
        sys: TrinitarianScaleRotorSystem,
        spec_readings: Optional[List[float]] = None,
        extra_laws: Sequence[AcademicLawCandidate] = (),
        calibrate_spectrometer: bool = False,
    ) -> None:
        sys.add_quantity(
            CoupledQuantity(
                "emission_wavelength_nm",
                600.0,
                ScaleTier.MECHANISM,
                "nm",
                "state",
                PinMode.FREE,
                50.0,
                20000.0,
            )
        )

        if spec_readings is not None:
            spec_offset = float(
                self.grafted_tools.get("spectrometer", {}).get("offset_deg", 0.0)
            )
            offset_mode = PinMode.FREE if calibrate_spectrometer else PinMode.FIXED_BOUNDARY
            sys.add_quantity(
                CoupledQuantity(
                    "spectrometer_offset_deg",
                    spec_offset,
                    ScaleTier.UPPER_ROTOR,
                    "deg",
                    "parameter",
                    offset_mode,
                    -45.0,
                    45.0,
                )
            )
            for idx, reading in enumerate(spec_readings):
                suf = "" if idx == 0 else f"_s{idx}"
                p_name = f"prism_angle_deg{suf}"
                sys.add_quantity(
                    CoupledQuantity(
                        p_name,
                        float(reading),
                        ScaleTier.FORMULA,
                        "deg",
                        "dial",
                        PinMode.DIAL,
                        10.0,
                        120.0,
                    )
                )
                sys.add_bond(
                    EqualityBond(
                        name=f"cauchy_spectrometer_law{suf}",
                        tier=ScaleTier.FORMULA,
                        variables=(p_name, "emission_wavelength_nm", "spectrometer_offset_deg"),
                        lhs_fn=lambda v, pn=p_name: (
                            (v[pn] - 30.0 - v["spectrometer_offset_deg"])
                            * ((v["emission_wavelength_nm"] / 500.0) ** 2)
                        )
                        / 18.0,
                        rhs_fn=lambda _v: 1.0,
                        lhs_expr=f"({p_name} - 30 - off) * (lambda/500)^2 / 18",
                        rhs_expr="1",
                    )
                )

        # Attach verified internalized disciplines + candidate laws (at step 0)
        for law in list(self.internalized_disciplines.values()) + list(extra_laws):
            bond = law.build_bond("")
            if all(v in sys.quantities for v in bond.variables):
                sys.add_bond(bond)

    def perceive_extended(
        self,
        world: ExtendedWorldState,
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
        disabled_channels: Sequence[str] = (),
        use_spectrometer: Optional[bool] = None,
        true_spectrometer_offset_deg: float = 0.0,
        extra_laws: Sequence[AcademicLawCandidate] = (),
    ) -> OrganismPerceptionState:
        """
        Full extended perception integrating biological organs, grafted
        spectrometer (if mounted), and verified internalized disciplines.
        """
        active_org_list, missing_org_list, obs_ch, miss_ch = self._active_and_missing(
            active_organs, disabled_channels
        )
        raw_imp = world.generate_sensory_impressions(
            body_pos_x=self.body_pos_x,
            body_vel_x=self.body_vel_x,
            eye_scale_bias=self.eye_scale_bias,
            eye_offset_bias=self.eye_offset_bias,
        )
        sys = self._build_coupled_nervous_system(
            steps=[(self.body_pos_x, self.body_vel_x, raw_imp, obs_ch)],
            calibrate_eye=False,
        )

        spec_active = self._spectrometer_is_active(use_spectrometer)
        spec_readings = None
        if spec_active:
            sr = world.generate_spectrometer_reading(
                spectrometer_offset_deg=true_spectrometer_offset_deg
            )
            spec_readings = [sr["prism_angle_deg"]]

        self._augment_nervous_system_with_extensions(
            sys=sys,
            spec_readings=spec_readings,
            extra_laws=extra_laws,
            calibrate_spectrometer=False,
        )

        rep = sys.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
            max_iter=240,
            tol=1e-11,
        )
        state = self._finalize_perception(
            sys=sys,
            active_organs=active_org_list,
            missing_organs=missing_org_list,
            observed_channels=obs_ch,
            missing_channels=miss_ch,
            action=rep.final_action,
        )
        state.inferred_world["emission_wavelength_nm"] = float(
            sys.quantities["emission_wavelength_nm"].value
        )
        if spec_readings is not None:
            state.reconstructed_organs["prism_angle_deg"] = float(spec_readings[0])
        return state

    def query_and_verify_knowledge_network(
        self,
        world: ExtendedWorldState,
        network: ExternalKnowledgeNetwork,
        target_vacuum_variable: str = "emission_wavelength_nm",
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
        true_spectrometer_offset_deg: float = 0.0,
        discover_workspace_keyword: Optional[str] = None,
    ) -> KnowledgeVerificationReport:
        """
        Queries external candidates for an unobservable vacuum variable, tests
        each claim inside counterfactual reality against active body/tool signals,
        rejects myths cleanly, and internalizes verified laws.
        """
        candidates = network.query(target_vacuum_variable)
        evaluated: List[Dict[str, Any]] = []
        accepted: List[str] = []
        rejected: List[str] = []

        spec_active = self._spectrometer_is_active(None)
        if not spec_active:
            raise RuntimeError(
                "Spectrometer must be actively measuring emission_wavelength_nm to falsify/verify external thermal radiation laws."
            )

        for cand in candidates:
            state = self.perceive_extended(
                world=world,
                active_organs=active_organs,
                use_spectrometer=True,
                true_spectrometer_offset_deg=true_spectrometer_offset_deg,
                extra_laws=(cand,),
            )
            action = float(state.total_action)
            ok = action < 1e-6
            if ok:
                self.internalized_disciplines[cand.name] = cand
                accepted.append(cand.name)
                reason = f"Empirically verified via Least Action (action={action:.2e})"
            else:
                rejected.append(cand.name)
                reason = f"Falsified by embodied sensory-tool conflict (action={action:.4f})"

            evaluated.append(
                {
                    "name": cand.name,
                    "source": cand.source,
                    "expression": cand.expression,
                    "empirical_action": action,
                    "accepted": ok,
                    "reason": reason,
                }
            )

        harvested_names: List[str] = []
        if discover_workspace_keyword:
            harvester = EpistemicToolHarvester(search_roots=[self.workspace_root])
            diag = DeficitDiagnosis(
                intent_text=discover_workspace_keyword,
                has_deficit=True,
                matched_pigments=[],
                missing_concepts=[discover_workspace_keyword],
                search_keywords=[discover_workspace_keyword],
                coverage_score=0.0,
                explanation_ko="외부 모듈 AST 탐색",
            )
            found = harvester.explore_external_tools(diag, max_candidates=5)
            harvested_names = [f"{c.module_name}.{c.class_name}" for c in found]

        self.cognitive_growth_rings.append(
            {
                "event": "verify_knowledge_network",
                "accepted": list(accepted),
                "rejected": list(rejected),
                "workspace_tools": list(harvested_names),
            }
        )
        return KnowledgeVerificationReport(
            query_vacuum_variable=target_vacuum_variable,
            evaluated_candidates=evaluated,
            accepted_laws=accepted,
            rejected_falsehoods=rejected,
            workspace_harvested_tools=harvested_names,
        )

    def drive_automobile_with_safety(
        self,
        world: ExtendedWorldState,
        desired_speed_ms: float,
        drive_duration_s: float = 2.0,
        active_organs: Sequence[str] = ("ear", "hand"),
        disabled_channels: Sequence[str] = ("touch_contact_force",),
        clearance_limit_m: Optional[float] = None,
    ) -> VehicleDriveReport:
        """
        Connects an automobile as an embodied motor extension governed by Layer-3
        braking & fuel Upper Rotor fences, then uses the regulated macroscopic
        displacement to collapse far-horizon parallax vacuums.
        """
        car_cfg = self.grafted_tools.get("automobile")
        if not car_cfg or not car_cfg.get("active", False):
            raise RuntimeError("No active automobile grafted; call graft_automobile() first.")

        friction_mu = float(car_cfg["friction_mu"])
        fuel_tank_l = float(car_cfg["fuel_tank_l"])
        fuel_rate = float(car_cfg["fuel_rate_l_per_m"])
        safety_margin = float(car_cfg["safety_margin_m"])
        limit_m = float(clearance_limit_m if clearance_limit_m is not None else world.source_pos_x)

        v_cmd = float(desired_speed_ms)
        dt = float(drive_duration_s)
        car_sys = TrinitarianScaleRotorSystem(name="automobile_motor_extension")
        car_sys.add_quantity(CoupledQuantity("body_pos_x", self.body_pos_x, ScaleTier.VALUE, "m", "cause", PinMode.FIXED_BOUNDARY, -500.0, 5000.0))
        car_sys.add_quantity(CoupledQuantity("drive_duration_s", dt, ScaleTier.VALUE, "s", "fixed", PinMode.FIXED_BOUNDARY, 0.1, 3600.0))
        car_sys.add_quantity(CoupledQuantity("friction_mu", friction_mu, ScaleTier.VALUE, "", "fixed", PinMode.FIXED_BOUNDARY, 0.05, 2.0))
        car_sys.add_quantity(CoupledQuantity("fuel_tank_l", fuel_tank_l, ScaleTier.UPPER_ROTOR, "L", "fence_metric", PinMode.FIXED_BOUNDARY, 0.1, 1000.0))
        car_sys.add_quantity(CoupledQuantity("safety_margin_m", safety_margin, ScaleTier.UPPER_ROTOR, "m", "fence_metric", PinMode.FIXED_BOUNDARY, 0.0, 500.0))
        car_sys.add_quantity(CoupledQuantity("clearance_limit_m", limit_m, ScaleTier.UPPER_ROTOR, "m", "fence_metric", PinMode.FIXED_BOUNDARY, 1.0, 10000.0))
        car_sys.add_quantity(CoupledQuantity("vehicle_speed_ms", v_cmd, ScaleTier.VALUE, "m/s", "dial", PinMode.DIAL, 0.0, 120.0))

        car_sys.add_quantity(CoupledQuantity("travel_m", v_cmd * dt, ScaleTier.FORMULA, "m", "process", PinMode.FREE, 0.0, 50000.0))
        car_sys.add_quantity(CoupledQuantity("body_pos_next", self.body_pos_x + v_cmd * dt, ScaleTier.MECHANISM, "m", "outcome", PinMode.FREE, -500.0, 50000.0))
        car_sys.add_quantity(CoupledQuantity("stopping_distance_m", (v_cmd ** 2) / (2.0 * friction_mu * GRAVITY_MS2), ScaleTier.MECHANISM, "m", "process", PinMode.FREE, 0.0, 50000.0))
        car_sys.add_quantity(CoupledQuantity("fuel_used_l", v_cmd * dt * fuel_rate, ScaleTier.MECHANISM, "L", "outcome", PinMode.FREE, 0.0, 1000.0))

        car_sys.add_bond(
            EqualityBond(
                name="kinematic_displacement",
                tier=ScaleTier.FORMULA,
                variables=("travel_m", "vehicle_speed_ms", "drive_duration_s"),
                lhs_fn=lambda v: v["travel_m"],
                rhs_fn=lambda v: v["vehicle_speed_ms"] * v["drive_duration_s"],
                lhs_expr="travel_m",
                rhs_expr="v * dt",
            )
        )
        car_sys.add_bond(
            EqualityBond(
                name="worldline_body_advance",
                tier=ScaleTier.MECHANISM,
                variables=("body_pos_next", "body_pos_x", "travel_m"),
                lhs_fn=lambda v: v["body_pos_next"],
                rhs_fn=lambda v: v["body_pos_x"] + v["travel_m"],
                lhs_expr="body_pos_next",
                rhs_expr="body_pos_x + travel_m",
            )
        )
        car_sys.add_bond(
            EqualityBond(
                name="kinetic_braking_law",
                tier=ScaleTier.MECHANISM,
                variables=("stopping_distance_m", "vehicle_speed_ms", "friction_mu"),
                lhs_fn=lambda v: v["stopping_distance_m"],
                rhs_fn=lambda v: (v["vehicle_speed_ms"] ** 2) / (2.0 * v["friction_mu"] * GRAVITY_MS2),
                lhs_expr="d_stop",
                rhs_expr="v^2 / (2 * mu * g)",
            )
        )
        car_sys.add_bond(
            EqualityBond(
                name="fuel_consumption_law",
                tier=ScaleTier.MECHANISM,
                variables=("fuel_used_l", "travel_m"),
                lhs_fn=lambda v, fr=fuel_rate: v["fuel_used_l"],
                rhs_fn=lambda v, fr=fuel_rate: v["travel_m"] * fr,
                lhs_expr="fuel_used_l",
                rhs_expr="travel_m * rate",
            )
        )
        car_sys.add_bond(
            EqualityBond(
                name="car_collision_safety_fence",
                tier=ScaleTier.UPPER_ROTOR,
                variables=("body_pos_next", "stopping_distance_m", "safety_margin_m", "clearance_limit_m"),
                lhs_fn=lambda v: v["body_pos_next"] + v["stopping_distance_m"] + v["safety_margin_m"],
                rhs_fn=lambda v: v["clearance_limit_m"],
                lhs_expr="pos_next + d_stop + margin",
                rhs_expr="clearance_limit",
                kind="fence_le",
                description="Stopping envelope must clear obstacle ahead",
            )
        )
        car_sys.add_bond(
            EqualityBond(
                name="car_fuel_fence",
                tier=ScaleTier.UPPER_ROTOR,
                variables=("fuel_used_l", "fuel_tank_l"),
                lhs_fn=lambda v: v["fuel_used_l"],
                rhs_fn=lambda v: v["fuel_tank_l"],
                lhs_expr="fuel_used_l",
                rhs_expr="fuel_tank_l",
                kind="fence_le",
                description="Fuel consumption bounded by tank capacity",
            )
        )

        car_sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        before_v = car_sys.observe_and_discern(adjustable_dials=["vehicle_speed_ms"])
        regulated = False
        if before_v.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION:
            _, after_v = car_sys.regulate_to_equilibrium(adjustable_dials=["vehicle_speed_ms"])
            regulated = True
        else:
            after_v = before_v

        reg_speed = float(car_sys.quantities["vehicle_speed_ms"].value)
        travel_m = float(car_sys.quantities["travel_m"].value)
        d_stop = float(car_sys.quantities["stopping_distance_m"].value)

        # Execute the real embodied motor step across the landscape!
        perc_after = self.probe_by_body_motion(
            world=world,
            delta_x=travel_m,
            active_organs=active_organs,
            disabled_channels=disabled_channels,
            true_eye_scale_bias=self.eye_scale_bias,
            true_eye_offset_bias=self.eye_offset_bias,
        )
        self.cognitive_growth_rings.append(
            {
                "event": "automobile_drive",
                "desired_speed": v_cmd,
                "regulated_speed": reg_speed,
                "travel_m": travel_m,
                "safety_regulated": regulated,
            }
        )
        return VehicleDriveReport(
            status=before_v.status,
            initial_speed_ms=v_cmd,
            regulated_speed_ms=reg_speed,
            travel_m=travel_m,
            stopping_distance_m=d_stop,
            clearance_m=limit_m,
            safety_regulated=regulated,
            perception_after_drive=perc_after,
            verdict_before=before_v,
            verdict_after=after_v,
        )

    def diagnose_and_calibrate_spectrometer(
        self,
        world: ExtendedWorldState,
        true_spectrometer_offset_deg: float = 5.0,
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
    ) -> ToolFaultReport:
        """
        Compares body organs + internalized disciplines against a drifting
        Spectrometer: detects conflict, proves body consensus, isolates the
        tool fault, and solves for the hidden prism offset without an oracle.
        """
        if not self._spectrometer_is_active(None):
            raise RuntimeError("Spectrometer must be mounted before diagnosing it.")

        # 1. Perceive with current spectrometer offset against body + internalized disciplines
        conflict_perc = self.perceive_extended(
            world=world,
            active_organs=active_organs,
            use_spectrometer=True,
            true_spectrometer_offset_deg=true_spectrometer_offset_deg,
        )
        conflict_before = float(conflict_perc.total_action)

        # 2. Check biological + internalized discipline consensus WITHOUT spectrometer
        body_only = self.perceive_extended(
            world=world,
            active_organs=active_organs,
            use_spectrometer=False,
        )
        body_action = float(body_only.total_action)

        if conflict_before <= 1e-4:
            return ToolFaultReport(
                tool_name="spectrometer",
                fault_detected=False,
                conflict_action_before=conflict_before,
                body_consensus_action=body_action,
                estimated_offset_deg=float(self.grafted_tools["spectrometer"]["offset_deg"]),
                recalibrated_action_after=conflict_before,
                detached_instead_of_calibrated=False,
                explanation="Spectrometer already aligned with embodied world consensus.",
            )

        # 3. Release spectrometer_offset_deg as FREE internal parameter under Least Action
        _, _, obs_ch, _ = self._active_and_missing(active_organs, ())
        raw_imp = world.generate_sensory_impressions(
            body_pos_x=self.body_pos_x,
            body_vel_x=self.body_vel_x,
            eye_scale_bias=self.eye_scale_bias,
            eye_offset_bias=self.eye_offset_bias,
        )
        sys_cal = self._build_coupled_nervous_system(
            steps=[(self.body_pos_x, self.body_vel_x, raw_imp, obs_ch)],
            calibrate_eye=False,
        )
        sr = world.generate_spectrometer_reading(spectrometer_offset_deg=true_spectrometer_offset_deg)
        self._augment_nervous_system_with_extensions(
            sys=sys_cal,
            spec_readings=[sr["prism_angle_deg"]],
            calibrate_spectrometer=True,
        )
        sys_cal.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
            max_iter=240,
            tol=1e-11,
        )
        est_offset = float(sys_cal.quantities["spectrometer_offset_deg"].value)
        self.grafted_tools["spectrometer"]["offset_deg"] = est_offset

        recalibrated = self.perceive_extended(
            world=world,
            active_organs=active_organs,
            use_spectrometer=True,
            true_spectrometer_offset_deg=true_spectrometer_offset_deg,
        )
        recal_action = float(recalibrated.total_action)

        self.cognitive_growth_rings.append(
            {
                "event": "recalibrate_spectrometer",
                "estimated_offset_deg": est_offset,
                "action_after": recal_action,
            }
        )
        return ToolFaultReport(
            tool_name="spectrometer",
            fault_detected=True,
            conflict_action_before=conflict_before,
            body_consensus_action=body_action,
            estimated_offset_deg=est_offset,
            recalibrated_action_after=recal_action,
            detached_instead_of_calibrated=False,
            explanation=(
                f"Detected spectrometer conflict (action={conflict_before:.4f}); "
                f"body+discipline consensus held (action={body_action:.2e}). "
                f"Least Action solved prism offset={est_offset:.5f} deg (residual action={recal_action:.2e})."
            ),
        )

    def sync_extended_to_causal_field(
        self,
        perc: OrganismPerceptionState,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "ext_org",
    ) -> Dict[str, Any]:
        """
        Projects both the biological organism AND grafted external tools +
        internalized academic disciplines into `CausalField` with uniform `(4,)`
        float32 voxel tensors, increasing semantic mass as cognition grows.
        """
        from core.physics.causal_field import InformationVoxel

        base_sync = perc.sync_to_causal_field(causal_field=causal_field, prefix=prefix)
        field = base_sync["causal_field"]
        vids = dict(base_sync["voxels"])

        spec_active = self._spectrometer_is_active(None)
        car_active = bool(self.grafted_tools.get("automobile", {}).get("active", False))
        n_laws = len(self.internalized_disciplines)

        ext_specs = [
            (
                "spectrometer_organ",
                [
                    1.0 if spec_active else 0.0,
                    perc.inferred_world.get("emission_wavelength_nm", 0.0) / 1000.0,
                    float(self.grafted_tools.get("spectrometer", {}).get("offset_deg", 0.0)) / 45.0,
                    perc.reconstructed_organs.get("prism_angle_deg", 0.0) / 90.0,
                ],
                [-1.8, 1.2, 1.0],
                1.2 if spec_active else 0.5,
            ),
            (
                "automobile_organ",
                [
                    1.0 if car_active else 0.0,
                    perc.body_pos_x / 50.0,
                    float(self.grafted_tools.get("automobile", {}).get("friction_mu", 0.0)),
                    float(self.grafted_tools.get("automobile", {}).get("safety_margin_m", 0.0)) / 20.0,
                ],
                [1.8, -1.0, -0.5],
                1.4 if car_active else 0.5,
            ),
            (
                "epistemic_disciplines",
                [
                    float(n_laws),
                    float(len(self.cognitive_growth_rings)) / 10.0,
                    perc.hangul_wave_coherence,
                    1.0 if n_laws > 0 else 0.0,
                ],
                [0.8, 1.6, 1.8],
                1.0 + 0.8 * n_laws,
            ),
        ]

        for key, vec4, pos3, mass_val in ext_specs:
            vid = f"{prefix}_{key}"
            vids[key] = vid
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"component": key, "growth_rings": len(self.cognitive_growth_rings)},
                    tensor=np.asarray(vec4, dtype=np.float32),
                    mass=float(mass_val),
                    position=np.asarray(pos3[: field.dimensions], dtype=np.float32),
                )
            )

        apex_id = vids["upper_self_rotor"]
        field.voxels[apex_id].mass += float(0.6 * len(self.grafted_tools) + 0.9 * n_laws)

        ext_links = [
            (vids["spectrometer_organ"], vids["world_core"], 0.9 if spec_active else 0.2),
            (vids["spectrometer_organ"], vids["vacuum_cavity"], 0.9 if spec_active else 0.2),
            (vids["automobile_organ"], vids["body_motor"], 0.95 if car_active else 0.2),
            (vids["epistemic_disciplines"], apex_id, 0.95 if n_laws > 0 else 0.3),
            (vids["epistemic_disciplines"], vids["vacuum_cavity"], 0.90 if n_laws > 0 else 0.3),
        ]
        for a, b, st in ext_links:
            field.link_voxels(a, b, strength=float(st))

        return {
            "causal_field": field,
            "voxels": vids,
            "beams_linked": base_sync["beams_linked"] + len(ext_links),
            "upper_rotor_mass": float(field.voxels[apex_id].mass),
        }
