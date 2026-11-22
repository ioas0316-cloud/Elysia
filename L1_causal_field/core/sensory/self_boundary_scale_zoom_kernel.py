r"""
self_boundary_scale_zoom_kernel.py

Scale Zoom-In / Zoom-Out Self-Boundary Kernel
=============================================
Embodiment of AGENTS.md §5 (Scale Boundary Layer) and §6 (4-Axis Self-Boundary Model):

Instead of holding every domain at fixed, brute-force full resolution or locking
into separate per-domain programs, ONE unified kernel dynamically shifts ELYSIA's
Scale Zoom (`zoom_scale` ∈ [0.0=Micro, 1.0=Meso, 2.0=Macro]) AND reshapes which
external entities count as part of her Extended Self vs External Environment:

  1. Agency Resonance -> Boundary Expansion (Zoom-Out into Tool / Avatar):
     Entities that synchronize closely with active intent enter `extended_self_ids`
     via reinforced `ConnectivityBeam` links inside `CausalField`.
  2. Severance / Cutscene -> Boundary Retraction (Zoom-Out Context Check):
     When previously extended control vanishes, ELYSIA tears that beam cleanly and
     zooms outward to `L_MACRO` to reassess external friction instead of hallucinating.
  3. Sharp Local Friction -> Pinpoint Zoom-In (e.g., Runtime Fault Isolation):
     When a local anomaly spikes inside a global stream (e.g. `ZeroDivisionError` in
     real AST-compiled Python code), `zoom_scale` focuses cleanly into `L_MICRO`
     while skipping healthy macroscopic chunks, saving compute.
  4. Passive Wide Context -> Lightweight Adaptive Sampling:
     On real external `.mp4` camera video (`real_external_nature_video.mp4`),
     `zoom_scale` sets actual sampling grid density (`32x24` / `64x48` / `96x64`),
     achieving substantial compute savings without losing structural awareness.
"""

from __future__ import annotations

import ast
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.autopoietic_organ_genesis import SafeASTSandbox


REAL_EXTERNAL_MP4_PATH = Path(
    r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4"
)
DEFAULT_ZOOM_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_self_boundary_scale_zoom_dashboard.png"
)


# ======================================================================
# 1. Domain-Agnostic Self-Boundary Contracts
# ======================================================================
@dataclass
class BoundaryEventPacket:
    """Domain-agnostic boundary telemetry packet fed into `step_boundary_cycle`."""

    domain_tag: str
    step_label: str
    intent_active: bool
    agency_resonance: Dict[str, float]
    local_anomaly_sharpness: float
    global_context_mismatch: float
    work_units_full: float
    work_units_executed: float


@dataclass
class SelfBoundarySnapshot:
    """Observable state of ELYSIA's self-boundary & 4-axis causal attribution at step t."""

    step_idx: int
    domain_tag: str
    step_label: str
    zoom_scale: float
    zoom_band: str
    extended_self_ids: Tuple[str, ...]
    external_world_ids: Tuple[str, ...]
    boundary_event: str
    axis_internal_origin: float
    axis_external_friction: float
    axis_boundary_ignorance: float
    axis_criteria_revision: float
    compute_saved_ratio: float


# ======================================================================
# 2. Persistent Scale Zoom-In / Zoom-Out Self-Boundary Kernel
# ======================================================================
class SelfBoundaryScaleZoomKernel:
    """
    Single domain-agnostic kernel operating on ONE persistent `CausalField`.
    Never inspects `packet.domain_tag` inside control decisions: responds strictly
    to continuous resonance, local friction sharpness, and global context mismatch.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=6)
        self.self_core_id = "ELYSIA_SELF_CORE"
        self.causal_field.add_voxel(
            InformationVoxel(
                id=self.self_core_id,
                content="0_value Core Self Anchor",
                tensor=np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                mass=3.0,
                position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
                scale_level=1.0,
            )
        )
        self.zoom_scale: float = 1.0  # 0.0=Micro .. 1.0=Meso .. 2.0=Macro
        self.extended_self_ids: set[str] = set()
        self.external_world_ids: set[str] = set()
        self.entity_beams: Dict[str, float] = {}
        self.history: List[SelfBoundarySnapshot] = []

    @staticmethod
    def _classify_zoom_band(zoom_scale: float) -> str:
        if zoom_scale < 0.55:
            return "L_MICRO"
        if zoom_scale < 1.25:
            return "L_MESO"
        return "L_MACRO"

    def _sync_entity_voxel_and_beam(self, entity_id: str, strength: float, in_self: bool) -> None:
        clamped = float(np.clip(strength, 0.02, 1.0))
        if entity_id not in self.causal_field.voxels:
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=entity_id,
                    content=f"Boundary Entity ({entity_id})",
                    tensor=np.array([clamped, 0.5, 0.5, 0.5, 0.5, 1.0 if in_self else 0.0], dtype=np.float32),
                    mass=1.2 if in_self else 0.8,
                    position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
                    scale_level=self.zoom_scale,
                )
            )
            self.causal_field.link_voxels(self.self_core_id, entity_id, strength=clamped)
            return

        vox = self.causal_field.voxels[entity_id]
        vox.tensor = np.array([clamped, 0.5, 0.5, 0.5, 0.5, 1.0 if in_self else 0.0], dtype=np.float32)
        vox.scale_level = float(self.zoom_scale)
        for beam in self.causal_field.beams:
            if (beam.source_id == self.self_core_id and beam.target_id == entity_id) or (
                beam.source_id == entity_id and beam.target_id == self.self_core_id
            ):
                beam.strength = clamped
                return
        self.causal_field.link_voxels(self.self_core_id, entity_id, strength=clamped)

    def step_boundary_cycle(self, packet: BoundaryEventPacket) -> SelfBoundarySnapshot:
        """
        Single continuous transition cycle across ANY sensory/control domain.
        Updates self-vs-world connectivity beams, computes zoom-in / zoom-out shift,
        and logs explicit 4-axis causal attribution (`AGENTS.md` §6).
        """
        step_idx = len(self.history)
        boundary_event = "STEADY_EQUILIBRIUM"
        expanded_any = False
        retracted_any = False

        # 1) Self-Boundary Plasticity across active entities
        for eid, raw_res in packet.agency_resonance.items():
            res = float(np.clip(raw_res, 0.0, 1.0))
            prev_beam = self.entity_beams.get(eid, 0.35)
            was_extended = eid in self.extended_self_ids

            if packet.intent_active:
                if was_extended and res < 0.25:
                    # Control severed -> rapid physical beam tear rather than slow lag
                    new_beam = max(0.03, prev_beam * 0.22)
                else:
                    new_beam = 0.55 * prev_beam + 0.45 * res
            else:
                new_beam = 0.85 * prev_beam + 0.15 * 0.30

            self.entity_beams[eid] = new_beam

            if new_beam >= 0.65:
                if not was_extended:
                    expanded_any = True
                self.extended_self_ids.add(eid)
                self.external_world_ids.discard(eid)
            elif was_extended and new_beam < 0.35:
                retracted_any = True
                self.extended_self_ids.discard(eid)
                self.external_world_ids.add(eid)
            elif eid not in self.extended_self_ids:
                self.external_world_ids.add(eid)

            self._sync_entity_voxel_and_beam(eid, new_beam, eid in self.extended_self_ids)

        if retracted_any:
            boundary_event = "BOUNDARY_RETRACTED_FROM_TOOL"
        elif expanded_any:
            boundary_event = "BOUNDARY_EXPANDED_TO_TOOL"

        # 2) Continuous Scale Zoom Shift (Micro <-> Meso <-> Macro)
        local_sharp = float(np.clip(packet.local_anomaly_sharpness, 0.0, 1.0))
        global_mis = float(np.clip(packet.global_context_mismatch, 0.0, 1.0))

        if retracted_any or (global_mis > 0.50 and local_sharp < 0.55):
            target_zoom = 1.75
            rate = 0.52
            if not retracted_any:
                boundary_event = "ZOOM_OUT_CONTEXT_REASSESSMENT"
        elif local_sharp > 0.55 and global_mis < 0.45:
            target_zoom = 0.14
            rate = 0.58
            boundary_event = "ZOOM_IN_LOCAL_ANOMALY_FOCUS"
        elif not packet.intent_active and local_sharp < 0.35:
            target_zoom = 1.55
            rate = 0.35
            if boundary_event == "STEADY_EQUILIBRIUM":
                boundary_event = "ZOOM_OUT_PASSIVE_SURVEY"
        else:
            target_zoom = 1.08
            rate = 0.38

        prev_zoom = self.zoom_scale
        self.zoom_scale = float(np.clip((1.0 - rate) * prev_zoom + rate * target_zoom, 0.0, 2.0))
        zoom_delta = abs(self.zoom_scale - prev_zoom)

        # Update Core voxel's own scale awareness inside CausalField
        core_vox = self.causal_field.voxels[self.self_core_id]
        core_vox.scale_level = self.zoom_scale

        # 3) Explicit 4-Axis Self-Boundary Attribution (AGENTS.md §6)
        mean_ext_res = (
            float(np.mean([self.entity_beams[eid] for eid in self.extended_self_ids]))
            if self.extended_self_ids
            else 0.0
        )
        axis_internal = float(np.clip(0.65 * (1.0 if packet.intent_active else 0.2) + 0.35 * mean_ext_res, 0.0, 1.0))
        axis_external = float(np.clip(max(local_sharp, global_mis), 0.0, 1.0))
        axis_ignorance = float(np.clip(global_mis if retracted_any or local_sharp > 0.6 else 0.10, 0.0, 1.0))
        axis_revision = float(np.clip(zoom_delta * 1.5 + (0.6 if retracted_any else 0.0), 0.0, 1.0))

        full_u = max(float(packet.work_units_full), 1e-9)
        exec_u = float(np.clip(packet.work_units_executed, 0.0, full_u))
        saved_ratio = float(np.clip(1.0 - exec_u / full_u, 0.0, 1.0))

        snap = SelfBoundarySnapshot(
            step_idx=step_idx,
            domain_tag=packet.domain_tag,
            step_label=packet.step_label,
            zoom_scale=self.zoom_scale,
            zoom_band=self._classify_zoom_band(self.zoom_scale),
            extended_self_ids=tuple(sorted(self.extended_self_ids)),
            external_world_ids=tuple(sorted(self.external_world_ids)),
            boundary_event=boundary_event,
            axis_internal_origin=axis_internal,
            axis_external_friction=axis_external,
            axis_boundary_ignorance=axis_ignorance,
            axis_criteria_revision=axis_revision,
            compute_saved_ratio=saved_ratio,
        )
        self.history.append(snap)
        return snap


# ======================================================================
# 3. Three Domain Adapters Feeding ONE Persistent Kernel
# ======================================================================
def run_tool_avatar_boundary_scenario(kernel: SelfBoundaryScaleZoomKernel) -> Dict[str, Any]:
    """
    Domain 1: Active control expands ELYSIA's self-boundary to enclose `avatar_tool`
    while separating `wall_env` and `npc_other` outside. Sudden cutscene / control
    severance tears the tool beam and zooms outward to `L_MACRO`.
    """
    beam_traces: Dict[str, List[float]] = {
        "avatar_tool": [],
        "wall_env": [],
        "npc_other": [],
    }
    snaps: List[SelfBoundarySnapshot] = []

    # Phase A: Synchronized embodied control (10 steps)
    for t in range(10):
        pkt = BoundaryEventPacket(
            domain_tag="tool_avatar",
            step_label=f"active_ctrl_{t}",
            intent_active=True,
            agency_resonance={"avatar_tool": 0.92, "wall_env": 0.02, "npc_other": 0.08},
            local_anomaly_sharpness=0.10,
            global_context_mismatch=0.05,
            work_units_full=100.0,
            work_units_executed=55.0,
        )
        s = kernel.step_boundary_cycle(pkt)
        snaps.append(s)
        for k in beam_traces:
            beam_traces[k].append(kernel.entity_beams[k])

    # Phase B: Control severed / cutscene takeover (6 steps)
    for t in range(6):
        pkt = BoundaryEventPacket(
            domain_tag="tool_avatar",
            step_label=f"severed_{t}",
            intent_active=True,
            agency_resonance={"avatar_tool": 0.04, "wall_env": 0.02, "npc_other": 0.12},
            local_anomaly_sharpness=0.25,
            global_context_mismatch=0.85,
            work_units_full=100.0,
            work_units_executed=35.0,
        )
        s = kernel.step_boundary_cycle(pkt)
        snaps.append(s)
        for k in beam_traces:
            beam_traces[k].append(kernel.entity_beams[k])

    return {
        "snapshots": snaps,
        "beam_traces": beam_traces,
    }


_BUGGY_MODULE_SRC = (
    "def add_scores(a, b):\n"
    "    total = a + b\n"
    "    return total\n"
    "\n"
    "def blend_vectors(x, y, w):\n"
    "    res = (1.0 - w) * x + w * y\n"
    "    return res\n"
    "\n"
    "def normalize_energy(vals):\n"
    "    s = float(np.sum(np.abs(vals))) + 1e-12\n"
    "    return vals / s\n"
    "\n"
    "def compute_ratio(num, den):\n"
    "    out = num / den\n"
    "    return out\n"
    "\n"
    "def summarize_window(arr):\n"
    "    return float(np.mean(arr))\n"
)

_PATCHED_MODULE_SRC = (
    "def add_scores(a, b):\n"
    "    total = a + b\n"
    "    return total\n"
    "\n"
    "def blend_vectors(x, y, w):\n"
    "    res = (1.0 - w) * x + w * y\n"
    "    return res\n"
    "\n"
    "def normalize_energy(vals):\n"
    "    s = float(np.sum(np.abs(vals))) + 1e-12\n"
    "    return vals / s\n"
    "\n"
    "def compute_ratio(num, den):\n"
    "    safe_den = den if abs(den) > 1e-12 else 1.0\n"
    "    out = num / safe_den\n"
    "    return out\n"
    "\n"
    "def summarize_window(arr):\n"
    "    return float(np.mean(arr))\n"
)


def run_code_execution_zoom_scenario(kernel: SelfBoundaryScaleZoomKernel) -> Dict[str, Any]:
    """
    Domain 2: Scans a 5-function Python codebase coarsely at L_MACRO, executes inside
    `SafeASTSandbox`, genuinely hits `ZeroDivisionError` in `compute_ratio`, zooms IN
    to L_MICRO purely onto the faulting AST subgraph, then zooms back OUT when patched.
    """
    tree_buggy = ast.parse(_BUGGY_MODULE_SRC)
    func_nodes = [n for n in tree_buggy.body if isinstance(n, ast.FunctionDef)]
    total_ast_nodes = sum(1 for _ in ast.walk(tree_buggy))
    header_work = float(len(func_nodes))

    snaps: List[SelfBoundarySnapshot] = []
    # Phase 1: Coarse Macro scan (reads top-level function headers only)
    for t in range(3):
        s = kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="code_ast",
                step_label=f"macro_scan_{t}",
                intent_active=False,
                agency_resonance={"code_sandbox": 0.45},
                local_anomaly_sharpness=0.08,
                global_context_mismatch=0.05,
                work_units_full=float(total_ast_nodes),
                work_units_executed=header_work,
            )
        )
        snaps.append(s)

    # Phase 2: Genuine runtime execution inside SafeASTSandbox -> ZeroDivisionError
    ns_buggy = SafeASTSandbox.compile_and_load(_BUGGY_MODULE_SRC)
    isolated_function: Optional[str] = None
    isolated_nodes = float(total_ast_nodes)
    try:
        ns_buggy["add_scores"](2.0, 3.0)
        ns_buggy["blend_vectors"](1.0, 2.0, 0.25)
        ns_buggy["compute_ratio"](10.0, 0.0)
    except ZeroDivisionError as exc:
        tb_frames = traceback.extract_tb(exc.__traceback__)
        fault_lineno = tb_frames[-1].lineno if tb_frames else -1
        for fn in func_nodes:
            end_ln = fn.end_lineno or fn.lineno
            if fn.lineno <= fault_lineno <= end_ln:
                isolated_function = fn.name
                isolated_nodes = float(sum(1 for _ in ast.walk(fn)))
                break

    assert isolated_function == "compute_ratio"

    for t in range(4):
        s = kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="code_ast",
                step_label=f"micro_fault_{t}",
                intent_active=True,
                agency_resonance={"code_sandbox": 0.72},
                local_anomaly_sharpness=0.95,
                global_context_mismatch=0.10,
                work_units_full=float(total_ast_nodes),
                work_units_executed=isolated_nodes,
            )
        )
        snaps.append(s)

    # Phase 3: Patched execution verification -> relaxes zoom outward
    ns_patched = SafeASTSandbox.compile_and_load(_PATCHED_MODULE_SRC)
    patched_val = ns_patched["compute_ratio"](10.0, 0.0)
    assert np.isfinite(patched_val)

    for t in range(4):
        s = kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="code_ast",
                step_label=f"healed_relax_{t}",
                intent_active=True,
                agency_resonance={"code_sandbox": 0.85},
                local_anomaly_sharpness=0.05,
                global_context_mismatch=0.02,
                work_units_full=float(total_ast_nodes),
                work_units_executed=header_work,
            )
        )
        snaps.append(s)

    return {
        "snapshots": snaps,
        "isolated_function": isolated_function,
        "total_ast_nodes": total_ast_nodes,
        "isolated_ast_nodes": isolated_nodes,
    }


def run_real_video_adaptive_zoom_scenario(
    kernel: SelfBoundaryScaleZoomKernel,
    video_path: Path = REAL_EXTERNAL_MP4_PATH,
    max_frames: int = 75,
) -> Dict[str, Any]:
    """
    Domain 3: Feeds real `.mp4` frames (`real_external_nature_video.mp4`) into the SAME
    kernel. The kernel's live `zoom_scale` picks coarse (`32x24`), medium (`64x48`), or
    fine (`96x64`) resolution BEFORE sampling, yielding real lightweight compute savings.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open external video: {video_path}")

    full_w, full_h = 96, 64
    full_pixels = float(full_w * full_h)
    prev_coarse: Optional[np.ndarray] = None
    snaps: List[SelfBoundarySnapshot] = []
    resolutions_used: List[int] = []
    frames_read = 0

    try:
        while frames_read < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames_read += 1

            # Choose actual spatial sampling resolution strictly from current `zoom_scale`
            if kernel.zoom_scale >= 1.25:
                rw, rh = 32, 24  # L_MACRO: 12.5% pixel budget
            elif kernel.zoom_scale >= 0.55:
                rw, rh = 64, 48  # L_MESO: 50% pixel budget
            else:
                rw, rh = full_w, full_h  # L_MICRO: 100% full fine inspection

            resolutions_used.append(rw * rh)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            sampled = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)

            gx = np.abs(np.diff(sampled, axis=1))
            gy = np.abs(np.diff(sampled, axis=0))
            peak_grad = float(max(np.max(gx), np.max(gy)))
            mean_grad = float(0.5 * (np.mean(gx) + np.mean(gy))) + 1e-6
            grad_ratio = peak_grad / mean_grad
            local_sharp = float(np.clip((grad_ratio - 6.0) / 14.0, 0.05, 0.48))

            coarse_ref = cv2.resize(sampled, (32, 24), interpolation=cv2.INTER_AREA)
            global_drift = (
                float(np.clip(np.mean(np.abs(coarse_ref - prev_coarse)) * 4.5, 0.0, 1.0))
                if prev_coarse is not None
                else 0.05
            )
            prev_coarse = coarse_ref

            s = kernel.step_boundary_cycle(
                BoundaryEventPacket(
                    domain_tag="real_video",
                    step_label=f"mp4_f{frames_read}",
                    intent_active=False,
                    agency_resonance={"external_camera_stream": 0.12},
                    local_anomaly_sharpness=local_sharp,
                    global_context_mismatch=global_drift,
                    work_units_full=full_pixels,
                    work_units_executed=float(rw * rh),
                )
            )
            snaps.append(s)
    finally:
        cap.release()

    saved_ratios = [s.compute_saved_ratio for s in snaps]
    return {
        "snapshots": snaps,
        "frames_processed": frames_read,
        "resolutions_used": resolutions_used,
        "mean_compute_saved": float(np.mean(saved_ratios)) if saved_ratios else 0.0,
    }


# ======================================================================
# 4. 4-Panel Dashboard Renderer
# ======================================================================
def render_self_boundary_zoom_dashboard(
    kernel: SelfBoundaryScaleZoomKernel,
    d1: Dict[str, Any],
    d2: Dict[str, Any],
    d3: Dict[str, Any],
    output_path: Path = DEFAULT_ZOOM_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.34, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.97)

    # Panel 1: Continuous Zoom Scale Across All 3 Sequential Domains
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    steps = np.arange(len(kernel.history))
    zooms = [s.zoom_scale for s in kernel.history]
    ax1.axhspan(0.0, 0.55, color="#ff7b72", alpha=0.10, label="L_MICRO Band (Pinpoint Focus)")
    ax1.axhspan(0.55, 1.25, color="#58a6ff", alpha=0.08, label="L_MESO Band (Tool / Interaction)")
    ax1.axhspan(1.25, 2.05, color="#7ee787", alpha=0.08, label="L_MACRO Band (Wide / Lightweight)")
    ax1.plot(steps, zooms, color="#ffa657", lw=2.2, label="Self-Boundary Zoom Scale")

    n1 = len(d1["snapshots"])
    n2 = len(d2["snapshots"])
    ax1.axvline(n1 - 0.5, color="#8b949e", ls=":", lw=1.2)
    ax1.axvline(n1 + n2 - 0.5, color="#8b949e", ls=":", lw=1.2)
    ax1.text(n1 * 0.5, 1.92, "D1\nTool", ha="center", color="#c9d1d9", fontsize=7.6)
    ax1.text(n1 + n2 * 0.5, 1.92, "D2\nCode", ha="center", color="#c9d1d9", fontsize=7.6)
    ax1.text(n1 + n2 + len(d3["snapshots"]) * 0.5, 1.92, "D3: Real Camera MP4\n(Lightweight Macro Survey)", ha="center", color="#c9d1d9", fontsize=7.6)

    ax1.set_ylim(0.0, 2.15)
    ax1.set_xlabel("Continuous Step Across Unified Kernel", color="#c9d1d9", fontsize=9)
    ax1.set_ylabel("Zoom Scale (0=Micro .. 2=Macro)", color="#c9d1d9", fontsize=9)
    ax1.set_title("One Kernel: Dynamic Self-Boundary Zoom Across Domains", color="#e6edf3", fontsize=10.2)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2)

    # Panel 2: Domain 1 — Agency Boundary Expansion & Severance Tear
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    bt = d1["beam_traces"]
    t1 = np.arange(len(bt["avatar_tool"]))
    ax2.axvspan(9.5, len(t1) - 0.5, color="#ff7b72", alpha=0.12, label="Control Severed (Cutscene)")
    ax2.axhline(0.65, color="#8b949e", ls="--", lw=1.1, label="Extended-Self Threshold (0.65)")
    ax2.plot(t1, bt["avatar_tool"], color="#7ee787", lw=2.3, label="avatar_tool (Expands -> Retracts)")
    ax2.plot(t1, bt["npc_other"], color="#d2a8ff", lw=1.6, label="npc_other (External)")
    ax2.plot(t1, bt["wall_env"], color="#ff7b72", lw=1.6, label="wall_env (External)")
    ax2.set_ylim(0.0, 1.24)
    ax2.set_xlabel("Tool Interaction Step", color="#c9d1d9", fontsize=9)
    ax2.set_ylabel("ELYSIA_SELF_CORE Beam Strength", color="#c9d1d9", fontsize=9)
    ax2.set_title("Domain 1: Tool Boundary Expansion & Cutscene Severance", color="#e6edf3", fontsize=10.2)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.3)

    # Panel 3: Domain 2 — Real Code AST Zoom & Compute Allocation
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    s2 = d2["snapshots"]
    t2 = np.arange(len(s2))
    z2 = [s.zoom_scale for s in s2]
    sv2 = [s.compute_saved_ratio for s in s2]
    ax3.plot(t2, z2, color="#58a6ff", lw=2.2, marker="o", ms=4.5, label="Zoom Scale (Macro -> Micro -> Meso)")
    ax3.plot(t2, sv2, color="#7ee787", lw=1.9, ls="--", label="Skipped AST Compute Ratio")
    ax3.axvspan(2.5, 6.5, color="#ff7b72", alpha=0.14, label=f"ZeroDivisionError in {d2['isolated_function']}()")
    ax3.set_ylim(0.0, 2.05)
    ax3.set_xlabel("Code Inspection Phase Step", color="#c9d1d9", fontsize=9)
    ax3.set_ylabel("Zoom Scale / Compute Saved Ratio", color="#c9d1d9", fontsize=9)
    ax3.set_title("Domain 2: Runtime Bug Triggers Pinpoint AST Zoom-In", color="#e6edf3", fontsize=10.2)
    ax3.tick_params(colors="#c9d1d9", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.3)

    # Panel 4: Domain 3 — Real Camera Video Adaptive Pixel Savings
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    s3 = d3["snapshots"]
    t3 = np.arange(len(s3))
    res_frac = [r / float(96 * 64) for r in d3["resolutions_used"]]
    saved3 = [s.compute_saved_ratio for s in s3]
    ax4.fill_between(t3, 0.0, saved3, color="#7ee787", alpha=0.28, label=f"Compute Saved (mean={d3['mean_compute_saved']*100:.1f}%)")
    ax4.plot(t3, saved3, color="#7ee787", lw=2.0)
    ax4.plot(t3, res_frac, color="#ffa657", lw=1.8, label="Active Pixel Budget Used (32x24 / 64x48 / 96x64)")
    ax4.set_ylim(0.0, 1.24)
    ax4.set_xlabel("Real Camera Video Frame (real_external_nature_video.mp4)", color="#c9d1d9", fontsize=9)
    ax4.set_ylabel("Fraction of Full 96x64 Work", color="#c9d1d9", fontsize=9)
    ax4.set_title("Domain 3: Adaptive Zoom Keeps Real Video Lightweight", color="#e6edf3", fontsize=10.2)
    ax4.tick_params(colors="#c9d1d9", labelsize=8)
    ax4.legend(loc="center right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.3)

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
