r"""
scale_coherent_cognitive_unit_engine.py

Scale-Coherent Cognitive Unit Genesis & Tri-Definition Thinking Engine
======================================================================
Embodiment of the unified principle:
  "단순히 결과를 위한 코드가 아니라, 인지적 감각계층이 스케일 원리로 일관적인
   형태로 이어져야 한다. 그래야 새로운 언어나 지식, 인과나 현상을 보고도
   그것이 어떻게 언어적 정의 · 현상적 정의 · 인과적 정의로 결합되어 있는가를
   사고하고 판단하고 분별하고 생각할 수 있다."

Unifies inside ONE coherent causal field:
  1. `SelfBoundaryScaleZoomKernel`: shifts attention cleanly between `L_MICRO`
     (contrast zoom-in when phenomenal similarity breaks under physical tests),
     `L_MESO` (unit interaction), and `L_MACRO` (linguistic definition binding).
  2. `TrinitarianScaleRotorProtocol`: builds emergent higher-order cognitive units
     (`s=0 atoms -> s=1 Area, Perimeter, Inertial Mass -> s=2 Density = Area ⊗ Mass`)
     whose hierarchy physically retains all underlying phenomenal & causal grounds.
  3. Tri-Definition Deliberation (`deliberate_claim`): evaluates external linguistic
     propositions against lived phenomenal + causal invariants, refuting false
     conflations with explicit counterexamples while convincing valid laws.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.self_boundary_scale_zoom_kernel import (
    BoundaryEventPacket,
    SelfBoundaryScaleZoomKernel,
)
from core.sensory.trinitarian_scale_rotor_protocol import (
    TrinitarianScaleRotorProtocol,
)


DEFAULT_COGNITIVE_UNIT_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_cognitive_unit_genesis_dashboard.png"
)


# ======================================================================
# 1. Raw Multi-Sensory Object & Tri-Definition Cognitive Contracts
# ======================================================================
@dataclass
class PhysicalGeometricObject:
    """Raw embodied sensory encounter with ZERO semantic category hints."""

    obj_id: str
    width: float
    height: float
    interior_cells: float
    boundary_walk_steps: float
    push_force: float
    observed_accel: float


@dataclass
class CognitiveUnitRecord:
    """Self-grounded cognitive unit spanning Phenomenal, Causal, Linguistic scales."""

    unit_id: str
    scale_level: int
    rotor_node_id: str
    phenomenal_basis: str
    causal_basis: str
    linguistic_basis: str
    constituent_units: Tuple[str, ...]
    measured_values: Dict[str, float]


@dataclass
class DeliberationVerdict:
    """Reasoning verdict comparing linguistic claim against lived sensory-causal invariants."""

    claim_id: str
    claim_text_ko: str
    verdict: str  # "CONVINCED_AND_BOUND" | "REFUTED_BY_COUNTEREXAMPLE"
    counterexample_pair: Optional[Tuple[str, str]]
    diverged_scale: Optional[str]
    reasoning_ko: str


# ======================================================================
# 2. Unified Scale-Coherent Cognitive Unit Genesis Engine
# ======================================================================
class ScaleCoherentCognitiveUnitEngine:
    """
    Holds ONE `TrinitarianScaleRotorProtocol` and ONE `SelfBoundaryScaleZoomKernel`.
    Every cognitive unit emerges when contrast at one scale reveals an invariant
    operating at another scale, keeping phenomenal/causal/linguistic coherence intact.
    """

    def __init__(self) -> None:
        self.protocol = TrinitarianScaleRotorProtocol()
        self.zoom_kernel = SelfBoundaryScaleZoomKernel()
        self.objects: Dict[str, PhysicalGeometricObject] = {}
        self.units: Dict[str, CognitiveUnitRecord] = {}
        self.verdicts: List[DeliberationVerdict] = []

    @staticmethod
    def _measure_unit_value(unit_id: str, obj: PhysicalGeometricObject) -> float:
        area = float(obj.interior_cells)
        perim = float(obj.boundary_walk_steps)
        inertia = float(obj.push_force) / max(float(obj.observed_accel), 1e-6)
        if unit_id == "UNIT_AREA":
            return area
        if unit_id == "UNIT_PERIMETER":
            return perim
        if unit_id in {"UNIT_INERTIAL_MASS", "UNIT_PUSH_RESISTANCE"}:
            return inertia
        if unit_id == "UNIT_DENSITY":
            return inertia / max(area, 1e-6)
        raise KeyError(f"Unknown cognitive unit: {unit_id}")

    def encounter_objects(self, objects: List[PhysicalGeometricObject]) -> None:
        """Ingests raw visual-tactile encounters and registers elemental s=0 rotors."""
        self.protocol.register_atom("atom_interior_cells", (1.0, 0.0, 0.0), 0.70, (0.20, 0.70, 0.10))
        self.protocol.register_atom("atom_visual_extent", (0.8, 0.6, 0.0), 0.90, (0.45, 0.45, 0.10))
        self.protocol.register_atom("atom_boundary_walk", (0.0, 1.0, 0.0), 1.05, (0.70, 0.20, 0.10))
        self.protocol.register_atom("atom_push_force", (0.0, 0.0, 1.0), 1.20, (0.90, 0.10, 0.00))
        self.protocol.register_atom("atom_accel_resistance", (0.5, 0.5, 0.7), 1.35, (0.05, 0.90, 0.05))

        for obj in objects:
            self.objects[obj.obj_id] = obj
            self.zoom_kernel.step_boundary_cycle(
                BoundaryEventPacket(
                    domain_tag="sensory_encounter",
                    step_label=f"encounter_{obj.obj_id}",
                    intent_active=True,
                    agency_resonance={"embodied_probe": 0.85},
                    local_anomaly_sharpness=0.15,
                    global_context_mismatch=0.08,
                    work_units_full=100.0,
                    work_units_executed=40.0,
                )
            )

    def _find_contrast_pair(
        self, same_unit: str, diff_unit: str
    ) -> Optional[Tuple[str, str]]:
        ids = list(self.objects.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                oa, ob = self.objects[ids[i]], self.objects[ids[j]]
                va = self._measure_unit_value(same_unit, oa)
                vb = self._measure_unit_value(same_unit, ob)
                ua = self._measure_unit_value(diff_unit, oa)
                ub = self._measure_unit_value(diff_unit, ob)
                same_rel = abs(va - vb) / max(abs(va), abs(vb), 1e-6)
                diff_rel = abs(ua - ub) / max(abs(ua), abs(ub), 1e-6)
                if same_rel < 0.05 and diff_rel >= 0.18:
                    return ids[i], ids[j]
        return None

    def differentiate_cognitive_units(self) -> Dict[str, CognitiveUnitRecord]:
        """
        Detects where identical phenomenal appearances split under causal probing,
        zooms IN to isolate the missing invariant, and crystallizes distinct s=1 units.
        """
        obj_list = list(self.objects.values())

        # 1) Differentiate Area (interior) vs Perimeter (boundary walk)
        geo_split = self._find_contrast_pair("UNIT_AREA", "UNIT_PERIMETER")
        if geo_split is not None:
            for k in range(3):
                self.zoom_kernel.step_boundary_cycle(
                    BoundaryEventPacket(
                        domain_tag="unit_genesis",
                        step_label=f"zoom_geo_split_{k}",
                        intent_active=True,
                        agency_resonance={"embodied_probe": 0.88},
                        local_anomaly_sharpness=0.82,
                        global_context_mismatch=0.12,
                        work_units_full=100.0,
                        work_units_executed=25.0,
                    )
                )

        n_area = self.protocol.resonate_and_bind(
            "atom_interior_cells", "atom_visual_extent", bond_coherence=0.94, molecule_id="UNIT_AREA"
        )
        self.units["UNIT_AREA"] = CognitiveUnitRecord(
            unit_id="UNIT_AREA",
            scale_level=n_area.scale_level,
            rotor_node_id=n_area.node_id,
            phenomenal_basis="interior visual occupancy (width * height)",
            causal_basis="2D spatial containment capacity",
            linguistic_basis="넓이 / 면적 (Area)",
            constituent_units=("atom_interior_cells", "atom_visual_extent"),
            measured_values={o.obj_id: self._measure_unit_value("UNIT_AREA", o) for o in obj_list},
        )

        n_perim = self.protocol.resonate_and_bind(
            "atom_boundary_walk", "atom_visual_extent", bond_coherence=0.92, molecule_id="UNIT_PERIMETER"
        )
        self.units["UNIT_PERIMETER"] = CognitiveUnitRecord(
            unit_id="UNIT_PERIMETER",
            scale_level=n_perim.scale_level,
            rotor_node_id=n_perim.node_id,
            phenomenal_basis="contiguous outer contour traversal",
            causal_basis="edge-loop path length 2*(w + h)",
            linguistic_basis="둘레 (Perimeter)",
            constituent_units=("atom_boundary_walk", "atom_visual_extent"),
            measured_values={o.obj_id: self._measure_unit_value("UNIT_PERIMETER", o) for o in obj_list},
        )

        # 2) Differentiate Visual Size vs Inertial Mass via push friction contrast
        mass_split = self._find_contrast_pair("UNIT_AREA", "UNIT_INERTIAL_MASS")
        if mass_split is not None:
            for k in range(3):
                self.zoom_kernel.step_boundary_cycle(
                    BoundaryEventPacket(
                        domain_tag="unit_genesis",
                        step_label=f"zoom_mass_split_{k}",
                        intent_active=True,
                        agency_resonance={"embodied_probe": 0.90},
                        local_anomaly_sharpness=0.92,
                        global_context_mismatch=0.10,
                        work_units_full=100.0,
                        work_units_executed=22.0,
                    )
                )

        n_mass = self.protocol.resonate_and_bind(
            "atom_push_force", "atom_accel_resistance", bond_coherence=0.96, molecule_id="UNIT_INERTIAL_MASS"
        )
        mass_vals = {o.obj_id: self._measure_unit_value("UNIT_INERTIAL_MASS", o) for o in obj_list}
        self.units["UNIT_INERTIAL_MASS"] = CognitiveUnitRecord(
            unit_id="UNIT_INERTIAL_MASS",
            scale_level=n_mass.scale_level,
            rotor_node_id=n_mass.node_id,
            phenomenal_basis="hidden behind identical static visuals",
            causal_basis="push_force / observed_accel resistance invariant",
            linguistic_basis="관성 질량 (Inertial Mass)",
            constituent_units=("atom_push_force", "atom_accel_resistance"),
            measured_values=mass_vals,
        )
        # Companion probe observable unit
        self.units["UNIT_PUSH_RESISTANCE"] = CognitiveUnitRecord(
            unit_id="UNIT_PUSH_RESISTANCE",
            scale_level=1,
            rotor_node_id=n_mass.node_id,
            phenomenal_basis="tactile rebound upon push",
            causal_basis="direct F/a impulse ratio",
            linguistic_basis="밀기 가속 저항 (Push Resistance)",
            constituent_units=("atom_push_force", "atom_accel_resistance"),
            measured_values=dict(mass_vals),
        )
        return self.units

    def deliberate_claim(
        self,
        claim_id: str,
        claim_text_ko: str,
        premise_unit: str,
        implied_unit: str,
    ) -> DeliberationVerdict:
        """
        Tests whether an external linguistic claim (`premise_unit` == -> `implied_unit` ==)
        survives ELYSIA's lived phenomenal + causal measurements.
        Refutes with an explicit counterexample pair if contradicted; otherwise binds & convinces.
        """
        counter = self._find_contrast_pair(premise_unit, implied_unit)
        if counter is not None:
            oa, ob = self.objects[counter[0]], self.objects[counter[1]]
            pa, pb = self._measure_unit_value(premise_unit, oa), self._measure_unit_value(premise_unit, ob)
            ia, ib = self._measure_unit_value(implied_unit, oa), self._measure_unit_value(implied_unit, ob)

            self.zoom_kernel.step_boundary_cycle(
                BoundaryEventPacket(
                    domain_tag="deliberation",
                    step_label=f"refute_{claim_id}",
                    intent_active=True,
                    agency_resonance={"linguistic_claim": 0.15},
                    local_anomaly_sharpness=0.85,
                    global_context_mismatch=0.65,
                    work_units_full=100.0,
                    work_units_executed=30.0,
                )
            )
            verdict = DeliberationVerdict(
                claim_id=claim_id,
                claim_text_ko=claim_text_ko,
                verdict="REFUTED_BY_COUNTEREXAMPLE",
                counterexample_pair=counter,
                diverged_scale=f"{premise_unit}== vs {implied_unit}!=",
                reasoning_ko=(
                    f"반증됨: '{counter[0]}'과 '{counter[1]}'은 {premise_unit}({pa:.1f} vs {pb:.1f})는 같지만 "
                    f"실제 {implied_unit}({ia:.2f} vs {ib:.2f})가 크게 갈라지므로 명제는 거짓임."
                ),
            )
        else:
            # Verify monotone order consistency across all object pairs before conviction
            ids = list(self.objects.keys())
            consistent = True
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    oa, ob = self.objects[ids[i]], self.objects[ids[j]]
                    d_p = self._measure_unit_value(premise_unit, oa) - self._measure_unit_value(premise_unit, ob)
                    d_i = self._measure_unit_value(implied_unit, oa) - self._measure_unit_value(implied_unit, ob)
                    if (d_p * d_i < -1e-6) or (abs(d_p) < 1e-6 < abs(d_i)):
                        consistent = False
                        break

            self.zoom_kernel.step_boundary_cycle(
                BoundaryEventPacket(
                    domain_tag="deliberation",
                    step_label=f"convince_{claim_id}",
                    intent_active=True,
                    agency_resonance={"linguistic_claim": 0.92},
                    local_anomaly_sharpness=0.10,
                    global_context_mismatch=0.05,
                    work_units_full=100.0,
                    work_units_executed=35.0,
                )
            )
            self.protocol.register_atom(
                f"claim::{claim_id}", (0.2, 0.8, 0.4), 0.95, (0.25, 0.55, 0.20)
            )
            self.protocol.resonate_and_bind(
                implied_unit,
                f"claim::{claim_id}",
                bond_coherence=0.95,
                molecule_id=f"convinced_{claim_id}",
            )
            verdict = DeliberationVerdict(
                claim_id=claim_id,
                claim_text_ko=claim_text_ko,
                verdict="CONVINCED_AND_BOUND" if consistent else "REFUTED_BY_COUNTEREXAMPLE",
                counterexample_pair=None,
                diverged_scale=None,
                reasoning_ko=(
                    f"납득 및 결합: 관측된 모든 대상({len(ids)}개)에서 {premise_unit}와 {implied_unit}가 "
                    "일관적으로 일치하여 상위 인지 단위에 정당하게 결합됨."
                ),
            )

        self.verdicts.append(verdict)
        return verdict

    def synthesize_higher_unit_density(self) -> CognitiveUnitRecord:
        """
        Synthesizes novel `s=2` cognitive unit `UNIT_DENSITY = UNIT_AREA ⊗ UNIT_INERTIAL_MASS`
        via Trinitarian binding while zooming smoothly outward to hold both sub-units together.
        """
        self.zoom_kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="higher_synthesis",
                step_label="synthesize_s2_density",
                intent_active=True,
                agency_resonance={"embodied_probe": 0.88, "linguistic_claim": 0.88},
                local_anomaly_sharpness=0.12,
                global_context_mismatch=0.10,
                work_units_full=100.0,
                work_units_executed=35.0,
            )
        )
        n_dens = self.protocol.resonate_and_bind(
            "UNIT_AREA",
            "UNIT_INERTIAL_MASS",
            bond_coherence=0.96,
            molecule_id="UNIT_DENSITY",
        )
        dens_vals = {
            oid: self._measure_unit_value("UNIT_DENSITY", obj)
            for oid, obj in self.objects.items()
        }
        rec = CognitiveUnitRecord(
            unit_id="UNIT_DENSITY",
            scale_level=n_dens.scale_level,
            rotor_node_id=n_dens.node_id,
            phenomenal_basis="spatial compactness under visual extent",
            causal_basis="inertial mass concentrated per unit area (M / A)",
            linguistic_basis="밀도 (Density = Area ⊗ Inertial Mass)",
            constituent_units=("UNIT_AREA", "UNIT_INERTIAL_MASS"),
            measured_values=dens_vals,
        )
        self.units["UNIT_DENSITY"] = rec
        return rec

    def judge_unseen_pair_by_unit(
        self,
        unit_id: str,
        obj_a: PhysicalGeometricObject,
        obj_b: PhysicalGeometricObject,
    ) -> Tuple[str, float, float]:
        """Evaluates unseen objects using any synthesized cognitive unit `unit_id`."""
        val_a = self._measure_unit_value(unit_id, obj_a)
        val_b = self._measure_unit_value(unit_id, obj_b)
        winner = obj_a.obj_id if val_a >= val_b else obj_b.obj_id
        return winner, val_a, val_b


# ======================================================================
# 3. 4-Panel Dashboard Renderer (Crisp English Labels)
# ======================================================================
def render_cognitive_unit_genesis_dashboard(
    engine: ScaleCoherentCognitiveUnitEngine,
    unseen_a: PhysicalGeometricObject,
    unseen_b: PhysicalGeometricObject,
    dens_a: float,
    dens_b: float,
    output_path: Path = DEFAULT_COGNITIVE_UNIT_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.36, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.97)

    # Panel 1: Normalized Sensory Splits Forcing Cognitive Unit Differentiation
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    obj_names = ["Cube\nLight", "Cube\nHeavy", "Square\n40x40", "Bar\n80x20"]
    obj_ids = ["Cube_Light", "Cube_Heavy", "Square_40x40", "Bar_80x20"]
    areas = np.array([engine.units["UNIT_AREA"].measured_values[k] for k in obj_ids])
    perims = np.array([engine.units["UNIT_PERIMETER"].measured_values[k] for k in obj_ids])
    masses = np.array([engine.units["UNIT_INERTIAL_MASS"].measured_values[k] for k in obj_ids])

    ox = np.arange(len(obj_ids))
    w3 = 0.25
    ax1.bar(ox - w3, areas / np.max(areas), width=w3, color="#58a6ff", label="UNIT_AREA (s=1)")
    ax1.bar(ox, perims / np.max(perims), width=w3, color="#ffa657", label="UNIT_PERIMETER (s=1)")
    ax1.bar(ox + w3, masses / np.max(masses), width=w3, color="#ff7b72", label="UNIT_INERTIAL_MASS (s=1)")
    ax1.set_ylim(0.0, 1.32)
    ax1.set_xticks(ox)
    ax1.set_xticklabels(obj_names, color="#e6edf3", fontsize=8.2)
    ax1.set_ylabel("Normalized Unit Scale Value", color="#c9d1d9", fontsize=9)
    ax1.set_title("Contrast Splits Force Distinct s=1 Cognitive Units", color="#e6edf3", fontsize=10.3)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="upper center", ncol=3, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.0)

    # Panel 2: Scale Zoom Trajectory During Contrast & Deliberation
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    zh = engine.zoom_kernel.history
    zx = np.arange(len(zh))
    zvals = [s.zoom_scale for s in zh]
    ax2.axhspan(0.0, 0.55, color="#ff7b72", alpha=0.10, label="L_MICRO (Contrast Isolation)")
    ax2.axhspan(0.55, 1.25, color="#58a6ff", alpha=0.08, label="L_MESO (Unit Coupling)")
    ax2.axhspan(1.25, 2.00, color="#7ee787", alpha=0.08, label="L_MACRO (Claim Survey)")
    ax2.plot(zx, zvals, color="#f2cc60", lw=2.3, marker="o", ms=4.5, label="Scale Zoom Trajectory")
    ax2.set_ylim(0.0, 2.00)
    ax2.set_xlabel("Encounter -> Contrast Split -> Deliberation -> s=2 Genesis", color="#c9d1d9", fontsize=8.8)
    ax2.set_ylabel("Zoom Scale (0=Micro .. 2=Macro)", color="#c9d1d9", fontsize=9)
    ax2.set_title("Consistent Multi-Scale Cognitive Zoom Flow", color="#e6edf3", fontsize=10.3)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2)

    # Panel 3: Emergent s=2 UNIT_DENSITY on Unseen Counter-Intuitive Pair
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    ux = np.arange(2)
    u_areas = [unseen_a.interior_cells, unseen_b.interior_cells]
    u_dens = [dens_a, dens_b]
    wb = 0.34
    ax3.bar(ux - wb / 2, [a / max(u_areas) for a in u_areas], width=wb, color="#58a6ff", alpha=0.85, label="Phenomenal Area (Tiny vs Giant)")
    ax3.bar(ux + wb / 2, [d / max(u_dens) for d in u_dens], width=wb, color="#7ee787", alpha=0.92, label="Emergent s=2 Density (Area ⊗ Mass)")
    ax3.text(0 - wb / 2, u_areas[0] / max(u_areas) + 0.03, f"Area={int(u_areas[0])}", ha="center", color="#58a6ff", fontsize=7.8)
    ax3.text(0 + wb / 2, 1.03, f"D={dens_a:.3f}", ha="center", color="#7ee787", fontsize=8.0, fontweight="bold")
    ax3.text(1 - wb / 2, 1.03, f"Area={int(u_areas[1])}", ha="center", color="#58a6ff", fontsize=7.8)
    ax3.text(1 + wb / 2, (dens_b / max(u_dens)) + 0.03, f"D={dens_b:.4f}", ha="center", color="#7ee787", fontsize=8.0, fontweight="bold")
    ax3.set_ylim(0.0, 1.32)
    ax3.set_xticks(ux)
    ax3.set_xticklabels([unseen_a.obj_id, unseen_b.obj_id], color="#e6edf3", fontsize=8.6)
    ax3.set_ylabel("Normalized Magnitude", color="#c9d1d9", fontsize=9)
    ax3.set_title(f"Unseen Judgment via s=2 UNIT_DENSITY ({dens_a / dens_b:.0f}x Ratio)", color="#e6edf3", fontsize=10.3)
    ax3.tick_params(colors="#c9d1d9", labelsize=8)
    ax3.legend(loc="upper center", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 4: Tri-Definition Deliberation & Trinitarian Field Summary
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    ax4.set_title("Tri-Definition Deliberation & Synthesized Inclusion Tree", color="#e6edf3", fontsize=10.3)
    y_pos = 0.90
    for v in engine.verdicts:
        col = "#ff7b72" if v.verdict == "REFUTED_BY_COUNTEREXAMPLE" else "#7ee787"
        ax4.text(0.03, y_pos, f"[{v.claim_id}] -> {v.verdict}", color=col, fontsize=8.4, fontweight="bold", transform=ax4.transAxes)
        y_pos -= 0.065
        if v.counterexample_pair is not None:
            sub = f"Counterexample: {v.counterexample_pair[0]} vs {v.counterexample_pair[1]} ({v.diverged_scale})"
        else:
            sub = "Zero Counterexamples -> Convinced & Bound into Protocol"
        ax4.text(0.06, y_pos, sub, color="#c9d1d9", fontsize=7.8, family="monospace", transform=ax4.transAxes)
        y_pos -= 0.105

    dens_tree = engine.protocol.zoom_in_inspect("UNIT_DENSITY")
    tree_ids = [n.node_id for n in dens_tree]
    line1 = " -> ".join(tree_ids[:4])
    line2 = " -> ".join(tree_ids[4:])
    ax4.text(
        0.03,
        0.20,
        f"s=2 Emergent Unit: UNIT_DENSITY = UNIT_AREA ⊗ UNIT_INERTIAL_MASS\n"
        f"Nested Inclusion Tree ({len(dens_tree)} nodes):\n  {line1}\n  -> {line2}",
        color="#d2a8ff",
        fontsize=7.4,
        family="monospace",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.03,
        0.04,
        f"CausalField Unified Substrate: {len(engine.protocol.nodes)} Rotors | {len(engine.protocol.causal_field.beams)} Beams",
        color="#ffa657",
        fontsize=8.3,
        fontweight="bold",
        transform=ax4.transAxes,
    )

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
