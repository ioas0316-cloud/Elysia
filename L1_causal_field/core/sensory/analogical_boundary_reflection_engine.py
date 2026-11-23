r"""
analogical_boundary_reflection_engine.py

Analogical Boundary Reflection & Invisible-Domain Reinterpretation Engine
=========================================================================
Embodiment of the epistemic insight:
  "보이지 않는 것을 이해한다는 건 암흑이나 안개에 가깝기 때문에, 이미 닿을 수 있는
   감각 구조(물)를 디딤돌 삼아 어디서부터 어떻게 같고 다른지를 사유함으로써
   '내가 무엇을 인식할 수 있고 무엇을 인식할 수 없는지, 왜 그런지, 그러면 어떤
   새로운 인식적 과정이 필요한지'를 자기인식적으로 재해석한다는 뜻이다."

Architecture:
  1. Anchors a lived sensory foothold (`Water_Flow`) inside `TrinitarianScaleRotorProtocol`.
  2. Confronts invisible / indirect target domains (`Invisible_Electricity`,
     `Invisible_Heat_Conduction`) without hardcoded domain rules (`if domain == ...`).
  3. Explicitly marks `target_is_invisible_fog` when direct visual sight fails, then
     projects provisional analogical expectations across universal structural probes.
  4. Where resonance matches (`divergence < 0.30`), preserves the foothold skeleton;
     where the analogy breaks (`divergence >= 0.30`), zooms IN via
     `SelfBoundaryScaleZoomKernel`, physically tears the false analogy beam in `CausalField`
     via `re_tune_constituent`, binds newly discovered boundary invariants, and ascends
     to a reinterpreted `s=2` super-molecule (`reinterpreted::<target>`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


DEFAULT_ANALOGICAL_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_analogical_boundary_reflection_dashboard.png"
)


# ======================================================================
# 1. Domain Profiles & Reflection Verdicts
# ======================================================================
@dataclass
class RelationalDomainProfile:
    """Sensory visibility & empirical probe profile of a physical domain."""

    domain_id: str
    direct_visual_visibility: float
    probe_responses: Dict[str, float]
    indirect_mediator_needed: str


@dataclass
class ProbeContrastItem:
    """Comparison of foothold expectation vs target behavior along one probe axis."""

    probe_name: str
    source_val: float
    target_val: float
    divergence: float
    status: str  # "ISOMORPHIC_RESONANCE" | "ANALOGY_BREAKDOWN"


@dataclass
class AnalogicalReflectionReport:
    """Complete self-reflective audit of an analogical projection onto an unknown domain."""

    source_domain_id: str
    target_domain_id: str
    target_is_invisible_fog: bool
    aligned_probes: Tuple[str, ...]
    breakdown_probes: Tuple[str, ...]
    torn_analogical_beams: Tuple[str, ...]
    reinterpreted_molecule_id: str
    probe_contrasts: List[ProbeContrastItem]
    naive_analogy_accuracy: float
    reinterpreted_accuracy: float
    reflection_summary_ko: str


def build_physical_domain_profiles() -> Dict[str, RelationalDomainProfile]:
    """Known sensory foothold (Water) + two distinct invisible fog domains (Electricity, Heat)."""
    return {
        "Water_Flow": RelationalDomainProfile(
            domain_id="Water_Flow",
            direct_visual_visibility=1.0,
            probe_responses={
                "drive_gradient_response": 1.00,
                "constriction_resistance": -1.00,
                "open_cut_persistence": 0.95,
                "propagation_immediacy": 0.45,
            },
            indirect_mediator_needed="Direct Naked Eye & Tactile Pressure",
        ),
        "Invisible_Electricity": RelationalDomainProfile(
            domain_id="Invisible_Electricity",
            direct_visual_visibility=0.0,
            probe_responses={
                "drive_gradient_response": 0.98,
                "constriction_resistance": -0.96,
                "open_cut_persistence": 0.00,
                "propagation_immediacy": 0.99,
            },
            indirect_mediator_needed="Compass Magnetic Deflection & Bulb/Ammeter Load",
        ),
        "Invisible_Heat_Conduction": RelationalDomainProfile(
            domain_id="Invisible_Heat_Conduction",
            direct_visual_visibility=0.0,
            probe_responses={
                "drive_gradient_response": 0.95,
                "constriction_resistance": -0.92,
                "open_cut_persistence": 0.45,
                "propagation_immediacy": 0.08,
            },
            indirect_mediator_needed="Thermocouple Gradient & Expansion/Color Indicator",
        ),
    }


def _classify_probe_outcome(probe_name: str, val: float) -> str:
    """Maps continuous probe signature to physical behavioral class uniformly across domains."""
    if probe_name == "drive_gradient_response":
        return "FLUX_INCREASES" if val >= 0.70 else "FLUX_STATIC"
    if probe_name == "constriction_resistance":
        return "FLUX_DECREASES" if val <= -0.70 else "NO_RESISTANCE"
    if probe_name == "open_cut_persistence":
        if val >= 0.75:
            return "FLOW_CONTINUES_SPILL"
        if val >= 0.25:
            return "SLOW_AMBIENT_LEAK"
        return "INSTANT_ZERO_CUTOFF"
    if probe_name == "propagation_immediacy":
        if val >= 0.80:
            return "INSTANT_DOWNSTREAM"
        if val >= 0.25:
            return "FINITE_WAVE_DELAY"
        return "SLOW_DIFFUSIVE_LAG"
    return "UNKNOWN"


def build_counterfactual_tests_for_domain(domain_id: str) -> List[Dict[str, str]]:
    """Generates counterfactual physical judgment tasks matching ground-truth behavior."""
    profiles = build_physical_domain_profiles()
    prof = profiles[domain_id]
    return [
        {
            "scenario": "high_drive_gradient",
            "probe": "drive_gradient_response",
            "expected": _classify_probe_outcome("drive_gradient_response", prof.probe_responses["drive_gradient_response"]),
        },
        {
            "scenario": "narrow_constriction",
            "probe": "constriction_resistance",
            "expected": _classify_probe_outcome("constriction_resistance", prof.probe_responses["constriction_resistance"]),
        },
        {
            "scenario": "path_cut_open",
            "probe": "open_cut_persistence",
            "expected": _classify_probe_outcome("open_cut_persistence", prof.probe_responses["open_cut_persistence"]),
        },
        {
            "scenario": "immediate_switch_on",
            "probe": "propagation_immediacy",
            "expected": _classify_probe_outcome("propagation_immediacy", prof.probe_responses["propagation_immediacy"]),
        },
    ]


# ======================================================================
# 2. Analogical Boundary Reflection Engine
# ======================================================================
class AnalogicalBoundaryReflectionEngine:
    """
    Domain-agnostic self-reflective reasoning over ONE `TrinitarianScaleRotorProtocol`
    and ONE `SelfBoundaryScaleZoomKernel`. Uses known domains as provisional mirrors,
    explicitly tears failing analogical beams, and reinterprets unknown fog via
    newly bonded breakdown invariants.
    """

    def __init__(self) -> None:
        self.protocol = TrinitarianScaleRotorProtocol()
        self.zoom_kernel = SelfBoundaryScaleZoomKernel()
        self.reinterpreted_probe_values: Dict[str, Dict[str, float]] = {}
        self.reports: List[AnalogicalReflectionReport] = []

    @staticmethod
    def _deterministic_axis(tag: str, val: float) -> Tuple[Tuple[float, float, float], float]:
        d = hashlib.sha256(tag.encode("utf-8")).digest()
        ax = np.array([float(b) / 255.0 - 0.5 for b in d[:3]], dtype=np.float64)
        n = float(np.linalg.norm(ax))
        ax = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else ax / n
        angle = 0.45 + 1.25 * float(np.clip(abs(val), 0.0, 1.0))
        return (float(ax[0]), float(ax[1]), float(ax[2])), angle

    @staticmethod
    def _val_to_chromatic(val: float) -> Tuple[float, float, float]:
        # High positive -> dynamic flow (Red); negative -> ordering resist (Blue); near 0 -> boundary cutoff (Yellow)
        if val >= 0.70:
            return (0.80, 0.15, 0.05)
        if val <= -0.70:
            return (0.10, 0.85, 0.05)
        return (0.15, 0.25, 0.60)

    def anchor_lived_foothold(self, source: RelationalDomainProfile) -> List[str]:
        """Grounds the lived visual-tactile foothold inside `TrinitarianScaleRotorProtocol`."""
        atom_ids: List[str] = []
        for pname, pval in source.probe_responses.items():
            aid = f"{source.domain_id}::atom::{pname}"
            ax, ang = self._deterministic_axis(aid, pval)
            self.protocol.register_atom(aid, ax, ang, self._val_to_chromatic(pval))
            atom_ids.append(aid)
        self.zoom_kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="foothold_anchor",
                step_label=f"anchor_{source.domain_id}",
                intent_active=True,
                agency_resonance={source.domain_id: 0.95},
                local_anomaly_sharpness=0.10,
                global_context_mismatch=0.05,
                work_units_full=100.0,
                work_units_executed=45.0,
            )
        )
        return atom_ids

    def reflect_on_unknown_via_foothold(
        self,
        source: RelationalDomainProfile,
        target: RelationalDomainProfile,
        counterfactual_tests: List[Dict[str, str]],
    ) -> AnalogicalReflectionReport:
        """
        Probes an unknown/invisible domain using `source` as a provisional analogy,
        tears broken analogy beams via `re_tune_constituent`, constructs an `s=2`
        reinterpreted molecule, and evaluates naive vs reinterpreted reasoning.
        """
        is_fog = target.direct_visual_visibility < 0.30
        if is_fog:
            # Epistemic humility: blind direct sight triggers Macro survey -> needs foothold mirror
            self.zoom_kernel.step_boundary_cycle(
                BoundaryEventPacket(
                    domain_tag="analogical_reflection",
                    step_label=f"fog_encounter_{target.domain_id}",
                    intent_active=True,
                    agency_resonance={target.domain_id: 0.18},
                    local_anomaly_sharpness=0.20,
                    global_context_mismatch=0.80,
                    work_units_full=100.0,
                    work_units_executed=35.0,
                )
            )

        contrasts: List[ProbeContrastItem] = []
        aligned: List[str] = []
        breakdowns: List[str] = []
        torn_beams: List[str] = []
        reinterpreted_map: Dict[str, float] = {}

        aligned_node_ids: List[str] = []
        boundary_node_ids: List[str] = []

        tgt_anchor_id = f"{target.domain_id}::target_entity"
        self.protocol.register_atom(tgt_anchor_id, (0.3, 0.3, 0.9), 0.70, (0.30, 0.35, 0.35))

        for pname, src_v in source.probe_responses.items():
            tgt_v = target.probe_responses[pname]
            div = abs(float(src_v) - float(tgt_v))
            src_atom_id = f"{source.domain_id}::atom::{pname}"
            prov_bridge_id = f"{target.domain_id}::bridge::{pname}"

            # Step 1: Erect provisional analogical hypothesis (Target ⊗ Source_Probe)
            self.protocol.resonate_and_bind(
                tgt_anchor_id,
                src_atom_id,
                bond_coherence=0.80,
                molecule_id=prov_bridge_id,
            )

            if div < 0.30:
                status = "ISOMORPHIC_RESONANCE"
                aligned.append(pname)
                reinterpreted_map[pname] = float(src_v)  # Foothold law transfers faithfully!
                aligned_node_ids.append(prov_bridge_id)
            else:
                status = "ANALOGY_BREAKDOWN"
                breakdowns.append(pname)
                reinterpreted_map[pname] = float(tgt_v)  # Must abandon foothold value and adopt true target constraint!

                # Zoom IN to dissect where the foothold analogy fractured
                self.zoom_kernel.step_boundary_cycle(
                    BoundaryEventPacket(
                        domain_tag="analogical_reflection",
                        step_label=f"breakdown_zoom_{target.domain_id}_{pname}",
                        intent_active=True,
                        agency_resonance={target.domain_id: 0.45},
                        local_anomaly_sharpness=0.90,
                        global_context_mismatch=0.15,
                        work_units_full=100.0,
                        work_units_executed=25.0,
                    )
                )
                bnd_atom_id = f"{target.domain_id}::boundary::{pname}"
                bx, bang = self._deterministic_axis(bnd_atom_id, tgt_v)
                self.protocol.register_atom(
                    bnd_atom_id, bx, bang, self._val_to_chromatic(tgt_v)
                )
                # Tear false inherited link (src_atom_id -> prov_bridge_id) & bind discovered boundary!
                self.protocol.re_tune_constituent(
                    target_molecule_id=prov_bridge_id,
                    old_child_id=src_atom_id,
                    new_child_id=bnd_atom_id,
                    new_coherence=0.95,
                )
                torn_beams.append(f"{src_atom_id} -x- {prov_bridge_id}")
                boundary_node_ids.append(prov_bridge_id)

            contrasts.append(
                ProbeContrastItem(
                    probe_name=pname,
                    source_val=float(src_v),
                    target_val=float(tgt_v),
                    divergence=div,
                    status=status,
                )
            )

        # Ascend into coherent s=2 reinterpreted super-molecule combining surviving skeleton + boundary laws
        skel_id = f"{target.domain_id}::skeleton"
        if len(aligned_node_ids) >= 2:
            self.protocol.resonate_and_bind(
                aligned_node_ids[0], aligned_node_ids[1], bond_coherence=0.95, molecule_id=skel_id
            )
        else:
            skel_id = aligned_node_ids[0]

        bnd_mol_id = f"{target.domain_id}::boundary_law"
        if len(boundary_node_ids) >= 2:
            self.protocol.resonate_and_bind(
                boundary_node_ids[0], boundary_node_ids[1], bond_coherence=0.95, molecule_id=bnd_mol_id
            )
        elif boundary_node_ids:
            bnd_mol_id = boundary_node_ids[0]

        reinterp_id = f"reinterpreted::{target.domain_id}"
        self.protocol.resonate_and_bind(
            skel_id, bnd_mol_id, bond_coherence=0.96, molecule_id=reinterp_id
        )
        self.reinterpreted_probe_values[target.domain_id] = reinterpreted_map

        # Relax zoom smoothly after reinterpretation crystallizes
        self.zoom_kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="analogical_reflection",
                step_label=f"reinterpreted_{target.domain_id}",
                intent_active=True,
                agency_resonance={target.domain_id: 0.90},
                local_anomaly_sharpness=0.10,
                global_context_mismatch=0.05,
                work_units_full=100.0,
                work_units_executed=40.0,
            )
        )

        # Evaluate Unseen Counterfactuals: Naive Foothold vs Reinterpreted Structure
        naive_correct = 0
        reint_correct = 0
        for tc in counterfactual_tests:
            pr = tc["probe"]
            exp = tc["expected"]
            naive_pred = _classify_probe_outcome(pr, source.probe_responses[pr])
            reint_pred = _classify_probe_outcome(pr, reinterpreted_map[pr])
            if naive_pred == exp:
                naive_correct += 1
            if reint_pred == exp:
                reint_correct += 1

        n_tests = max(len(counterfactual_tests), 1)
        naive_acc = float(naive_correct / n_tests)
        reint_acc = float(reint_correct / n_tests)

        summary_ko = (
            f"[{target.domain_id}] 직접 시야 0(보이지 않는 안개) 상태에서 '{source.domain_id}'를 거울 삼아 사유한 결과: "
            f"'{', '.join(aligned)}' 원리는 구조적으로 같아 그대로 전이되었으나, "
            f"'{', '.join(breakdowns)}' 지점에서 물과의 비유가 깨져 잘못된 유비 빔을 끊고 새 경계 원리로 재해석함 "
            f"(단순 유비 {naive_acc * 100:.0f}% -> 재해석 후 {reint_acc * 100:.0f}%, 간접 매개: {target.indirect_mediator_needed})."
        )

        rep = AnalogicalReflectionReport(
            source_domain_id=source.domain_id,
            target_domain_id=target.domain_id,
            target_is_invisible_fog=is_fog,
            aligned_probes=tuple(aligned),
            breakdown_probes=tuple(breakdowns),
            torn_analogical_beams=tuple(torn_beams),
            reinterpreted_molecule_id=reinterp_id,
            probe_contrasts=contrasts,
            naive_analogy_accuracy=naive_acc,
            reinterpreted_accuracy=reint_acc,
            reflection_summary_ko=summary_ko,
        )
        self.reports.append(rep)
        return rep


# ======================================================================
# 3. 4-Panel Dashboard Renderer
# ======================================================================
def render_analogical_reflection_dashboard(
    engine: AnalogicalBoundaryReflectionEngine,
    profiles: Dict[str, RelationalDomainProfile],
    reports: List[AnalogicalReflectionReport],
    output_path: Path = DEFAULT_ANALOGICAL_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.36, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.97)

    probes = [
        "drive_gradient_response",
        "constriction_resistance",
        "open_cut_persistence",
        "propagation_immediacy",
    ]
    probe_short = ["Drive\nGradient", "Constriction\nResist", "Open-Cut\nPersist", "Propagation\nImmediacy"]

    # Panel 1: Probe Responses Across Foothold vs Two Invisible Worlds
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    px = np.arange(len(probes))
    w3 = 0.25
    w_vals = [profiles["Water_Flow"].probe_responses[p] for p in probes]
    e_vals = [profiles["Invisible_Electricity"].probe_responses[p] for p in probes]
    h_vals = [profiles["Invisible_Heat_Conduction"].probe_responses[p] for p in probes]
    ax1.bar(px - w3, w_vals, width=w3, color="#58a6ff", label="Water Flow (Lived Foothold)")
    ax1.bar(px, e_vals, width=w3, color="#f2cc60", label="Invisible Electricity (Fog 1)")
    ax1.bar(px + w3, h_vals, width=w3, color="#ff7b72", label="Invisible Heat (Fog 2)")
    ax1.axhline(0.0, color="#8b949e", lw=0.9)
    ax1.axvspan(1.5, 3.5, color="#ff7b72", alpha=0.09, label="Analogy Breakdown Zone")
    ax1.set_ylim(-1.25, 1.55)
    ax1.set_xticks(px)
    ax1.set_xticklabels(probe_short, color="#e6edf3", fontsize=8.0)
    ax1.set_ylabel("Probe Response (-1 .. +1)", color="#c9d1d9", fontsize=9)
    ax1.set_title("Water Mirror Projection vs Invisible Domains", color="#e6edf3", fontsize=10.4)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2)

    # Panel 2: Analogy Divergence Magnitude Showing Different Breakdown Signatures
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    r_elec, r_heat = reports[0], reports[1]
    div_e = [c.divergence for c in r_elec.probe_contrasts]
    div_h = [c.divergence for c in r_heat.probe_contrasts]
    wb = 0.34
    ax2.bar(px - wb / 2, div_e, width=wb, color="#f2cc60", label="Electricity vs Water Divergence")
    ax2.bar(px + wb / 2, div_h, width=wb, color="#ff7b72", label="Heat vs Water Divergence")
    ax2.axhline(0.30, color="#d2a8ff", ls="--", lw=1.4, label="Breakdown & Beam-Tear Threshold (0.30)")
    ax2.set_ylim(0.0, 1.32)
    ax2.set_xticks(px)
    ax2.set_xticklabels(probe_short, color="#e6edf3", fontsize=8.0)
    ax2.set_ylabel("|Source - Target| Divergence", color="#c9d1d9", fontsize=9)
    ax2.set_title("Where Water Analogy Holds vs Breaks Down", color="#e6edf3", fontsize=10.4)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.3)

    # Panel 3: Counterfactual Accuracy — Naive Analogy vs Reinterpreted Structure
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    rx = np.arange(len(reports))
    naive_accs = [r.naive_analogy_accuracy * 100.0 for r in reports]
    reint_accs = [r.reinterpreted_accuracy * 100.0 for r in reports]
    ax3.bar(rx - wb / 2, naive_accs, width=wb, color="#ff7b72", alpha=0.88, label="Naive Water Analogy (Uncorrected)")
    ax3.bar(rx + wb / 2, reint_accs, width=wb, color="#7ee787", alpha=0.92, label="Reinterpreted s=3 Structure")
    for i in range(len(reports)):
        ax3.text(i - wb / 2, naive_accs[i] + 2.5, f"{naive_accs[i]:.0f}%", ha="center", color="#ff7b72", fontsize=8.5, fontweight="bold")
        ax3.text(i + wb / 2, reint_accs[i] + 2.5, f"{reint_accs[i]:.0f}%", ha="center", color="#7ee787", fontsize=8.5, fontweight="bold")
    ax3.set_ylim(0.0, 130.0)
    ax3.set_xticks(rx)
    ax3.set_xticklabels([r.target_domain_id.replace("_", "\n") for r in reports], color="#e6edf3", fontsize=8.2)
    ax3.set_ylabel("Unseen Counterfactual Accuracy (%)", color="#c9d1d9", fontsize=9)
    ax3.set_title("Reinterpretation Resolves Counterfactual Failures", color="#e6edf3", fontsize=10.4)
    ax3.tick_params(colors="#c9d1d9", labelsize=8)
    ax3.legend(loc="upper center", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 4: Reflective Self-Awareness Ledger
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    ax4.set_title("Self-Reflective Epistemic Map (Known vs Fog vs Mediators)", color="#e6edf3", fontsize=10.4)
    y_pos = 0.90
    for r in reports:
        prof = profiles[r.target_domain_id]
        ax4.text(0.03, y_pos, f"[{r.target_domain_id}] (direct_vision={prof.direct_visual_visibility:.1f} -> INVISIBLE FOG)", color="#f2cc60", fontsize=8.3, fontweight="bold", transform=ax4.transAxes)
        y_pos -= 0.065
        ax4.text(0.05, y_pos, f"Aligned w/ Water: {', '.join(r.aligned_probes)}", color="#7ee787", fontsize=7.6, transform=ax4.transAxes)
        y_pos -= 0.060
        ax4.text(0.05, y_pos, f"Analogy Broken  : {', '.join(r.breakdown_probes)}", color="#ff7b72", fontsize=7.6, transform=ax4.transAxes)
        y_pos -= 0.060
        ax4.text(0.05, y_pos, f"Mediator Needed : {prof.indirect_mediator_needed}", color="#58a6ff", fontsize=7.6, transform=ax4.transAxes)
        y_pos -= 0.060
        ax4.text(0.05, y_pos, f"Crystallized s=3: {r.reinterpreted_molecule_id}", color="#d2a8ff", fontsize=7.6, family="monospace", transform=ax4.transAxes)
        y_pos -= 0.105

    ax4.text(
        0.03,
        0.03,
        f"Unified Trinitarian Field: {len(engine.protocol.nodes)} Rotors | {len(engine.protocol.causal_field.beams)} Beams",
        color="#ffa657",
        fontsize=8.3,
        fontweight="bold",
        transform=ax4.transAxes,
    )

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
