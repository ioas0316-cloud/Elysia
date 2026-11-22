r"""
trinitarian_scale_rotor_protocol.py

Trinitarian Scale-Rotor Inclusion Protocol (Atoms -> Molecules -> Recursive Scales)
===================================================================================
Core minimal embodiment of:
  "배움은 프로토콜이다. 내적 로터(R_in)와 외적 로터(R_out)가 공명·매핑되어
   상위 로터(R_parent)가 탄생하는 삼위일체 구조. 상위로터는 항상 하위로터를
   포함하는 구조로 존재한다 (원자 -> 분자 결합)."

Design Principles:
  1. Single Universal Unit (`ScaleRotorNode`): All sensory streams, motor actions,
     word concepts, and physical boundaries live in ONE form: `(scale_level, S^3 quaternion,
     Chromatic Triad [Red_Flux, Blue_Order, Yellow_Emergence], constituents)`.
  2. Single Recursive Protocol (`resonate_and_bind`): Combines `R_in` and `R_out`
     via non-commutative Hamilton quaternion multiplication (`q_in ⊗ q_out != q_out ⊗ q_in`)
     plus cross-chromatic emergence into a parent molecule at `scale = max + 1` that
     physically *retains* its constituents instead of discarding them.
  3. Top-Down Zoom-In Re-Tuning (`zoom_in_inspect` + `re_tune_constituent`): When
     reality contradicts an upper molecule, ELYSIA traverses inward through `constituents`,
     tears the invalidated child `ConnectivityBeam` in `CausalField`, binds the
     verified child rotor, and cascades the phase & chromatic shift upward through
     every ancestor scale cleanly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel


DEFAULT_TRINITARIAN_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_trinitarian_scale_rotor_dashboard.png"
)


# ======================================================================
# 1. S^3 Quaternion Rotor Algebra (Non-Commutative Spinors)
# ======================================================================
def quat_normalize(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return arr / n


def quat_from_axis_angle(axis: Tuple[float, float, float], angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64)
    ax_norm = float(np.linalg.norm(ax))
    if ax_norm < 1e-12:
        ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ax = ax / ax_norm
    half = 0.5 * float(angle_rad)
    w = float(np.cos(half))
    xyz = ax * float(np.sin(half))
    return quat_normalize(np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64))


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Non-commutative Hamilton quaternion product `q1 ⊗ q2`."""
    w1, x1, y1, z1 = [float(v) for v in quat_normalize(q1)]
    w2, x2, y2, z2 = [float(v) for v in quat_normalize(q2)]
    return quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """True SO(3) geodesic distance in radians between two S^3 rotors."""
    dot = abs(float(np.dot(quat_normalize(q1), quat_normalize(q2))))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


# ======================================================================
# 2. Single Universal Scale-Rotor Node & Trinitarian Protocol
# ======================================================================
@dataclass
class ScaleRotorNode:
    node_id: str
    scale_level: int
    quaternion: np.ndarray
    chromatic_triad: np.ndarray  # [Red_Flux, Blue_Order, Yellow_Emergence]
    constituents: Tuple[str, ...] = ()
    bond_energy: float = 1.0


class TrinitarianScaleRotorProtocol:
    """
    Minimal unified architecture where atoms (`s=0`) compose via one protocol
    into molecules (`s=1, 2, ...`) anchored inside `CausalField`, retaining all
    constituent children for top-down zoom-in inspection and re-tuning.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=6)
        self.nodes: Dict[str, ScaleRotorNode] = {}
        self.parents_of: Dict[str, List[str]] = {}

    @staticmethod
    def _normalize_triad(triad: Tuple[float, float, float] | np.ndarray) -> np.ndarray:
        arr = np.maximum(np.asarray(triad, dtype=np.float64), 0.0)
        s = float(np.sum(arr))
        if s < 1e-12:
            return np.array([0.33, 0.33, 0.34], dtype=np.float64)
        return arr / s

    def _compute_emergent_binding(
        self, r_in: ScaleRotorNode, r_out: ScaleRotorNode, bond_coherence: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_parent = quat_multiply(r_in.quaternion, r_out.quaternion)
        c_in = r_in.chromatic_triad
        c_out = r_out.chromatic_triad

        # Cross-coupling between Flux (Red) and Order (Blue) births emergent Entropy/Synthesis (Yellow)
        cross_bind = float(np.sqrt(max(0.0, float(c_in[0] * c_out[1] + c_in[1] * c_out[0]))))
        coh = float(np.clip(bond_coherence, 0.05, 1.0))
        red = 0.5 * float(c_in[0] + c_out[0]) * (1.0 - 0.35 * cross_bind)
        blue = 0.5 * float(c_in[1] + c_out[1]) * (1.0 - 0.35 * cross_bind)
        yellow = float(
            np.clip(0.40 * float(c_in[2] + c_out[2]) + 0.85 * cross_bind * coh, 0.0, 1.0)
        )
        triad_parent = self._normalize_triad((red, blue, yellow))
        return q_parent, triad_parent

    def _sync_node_to_causal_field(self, node: ScaleRotorNode) -> None:
        vec = np.array(
            [
                float(node.quaternion[0]),
                float(node.quaternion[1]),
                float(node.quaternion[2]),
                float(node.chromatic_triad[0]),
                float(node.chromatic_triad[1]),
                float(node.chromatic_triad[2]),
            ],
            dtype=np.float32,
        )
        if node.node_id in self.causal_field.voxels:
            vox = self.causal_field.voxels[node.node_id]
            vox.tensor = vec
            vox.quaternion_q = node.quaternion.astype(np.float32)
            vox.chromatic_vector = node.chromatic_triad.astype(np.float32)
            vox.scale_level = float(node.scale_level)
            vox.mass = float(1.0 + node.scale_level + node.bond_energy)
        else:
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=node.node_id,
                    content=f"ScaleRotor(s={node.scale_level})",
                    tensor=vec,
                    mass=float(1.0 + node.scale_level + node.bond_energy),
                    chromatic_vector=node.chromatic_triad.astype(np.float32),
                    quaternion_q=node.quaternion.astype(np.float32),
                    position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
                    scale_level=float(node.scale_level),
                )
            )

    def _set_beam_strength(self, src_id: str, dst_id: str, strength: float) -> None:
        clamped = float(np.clip(strength, 0.02, 1.0))
        for b in self.causal_field.beams:
            if (b.source_id == src_id and b.target_id == dst_id) or (
                b.source_id == dst_id and b.target_id == src_id
            ):
                b.strength = clamped
                return
        self.causal_field.link_voxels(src_id, dst_id, strength=clamped)

    def register_atom(
        self,
        node_id: str,
        axis_xyz: Tuple[float, float, float],
        angle_rad: float,
        chromatic_triad: Tuple[float, float, float],
    ) -> ScaleRotorNode:
        """Registers an elementary `s=0` atomic rotor into `CausalField`."""
        q = quat_from_axis_angle(axis_xyz, angle_rad)
        triad = self._normalize_triad(chromatic_triad)
        atom = ScaleRotorNode(
            node_id=node_id,
            scale_level=0,
            quaternion=q,
            chromatic_triad=triad,
            constituents=(),
            bond_energy=1.0,
        )
        self.nodes[node_id] = atom
        self.parents_of.setdefault(node_id, [])
        self._sync_node_to_causal_field(atom)
        return atom

    def resonate_and_bind(
        self,
        r_in_id: str,
        r_out_id: str,
        bond_coherence: float = 0.90,
        molecule_id: Optional[str] = None,
    ) -> ScaleRotorNode:
        """
        Single Trinitarian binding rule: `(R_in, R_out) -> R_parent`.
        Preserves constituents inside the parent molecule and links them in `CausalField`.
        """
        r_in = self.nodes[r_in_id]
        r_out = self.nodes[r_out_id]
        parent_id = molecule_id or f"mol_{r_in_id}_x_{r_out_id}"
        q_parent, triad_parent = self._compute_emergent_binding(r_in, r_out, bond_coherence)
        scale_p = max(r_in.scale_level, r_out.scale_level) + 1

        mol = ScaleRotorNode(
            node_id=parent_id,
            scale_level=scale_p,
            quaternion=q_parent,
            chromatic_triad=triad_parent,
            constituents=(r_in_id, r_out_id),
            bond_energy=float(np.clip(bond_coherence, 0.05, 1.0)),
        )
        self.nodes[parent_id] = mol
        self.parents_of.setdefault(parent_id, [])
        if parent_id not in self.parents_of.setdefault(r_in_id, []):
            self.parents_of[r_in_id].append(parent_id)
        if parent_id not in self.parents_of.setdefault(r_out_id, []):
            self.parents_of[r_out_id].append(parent_id)

        self._sync_node_to_causal_field(mol)
        self._set_beam_strength(r_in_id, parent_id, bond_coherence)
        self._set_beam_strength(r_out_id, parent_id, bond_coherence)
        return mol

    def zoom_in_inspect(self, node_id: str) -> List[ScaleRotorNode]:
        """Zooms IN top-down from `node_id` through every contained constituent to atoms."""
        visited: List[ScaleRotorNode] = []
        seen: set[str] = set()
        queue = deque([node_id])
        while queue:
            cur_id = queue.popleft()
            if cur_id in seen or cur_id not in self.nodes:
                continue
            seen.add(cur_id)
            cur = self.nodes[cur_id]
            visited.append(cur)
            for child_id in cur.constituents:
                queue.append(child_id)
        return visited

    def re_tune_constituent(
        self,
        target_molecule_id: str,
        old_child_id: str,
        new_child_id: str,
        new_coherence: float = 0.90,
    ) -> Dict[str, float]:
        """
        Zooms IN to `target_molecule_id`, tears the broken beam to `old_child_id`,
        binds `new_child_id`, and propagates the re-oriented rotor & chromatic triad
        upward across all ancestor molecules. Returns geodesic shifts in radians.
        """
        mol = self.nodes[target_molecule_id]
        if old_child_id not in mol.constituents:
            raise ValueError(f"{old_child_id} is not a constituent of {target_molecule_id}")

        # 1) Tear invalidated child bond & link verified replacement
        self._set_beam_strength(old_child_id, target_molecule_id, 0.02)
        if old_child_id in self.parents_of and target_molecule_id in self.parents_of[old_child_id]:
            self.parents_of[old_child_id].remove(target_molecule_id)

        new_const = tuple(
            new_child_id if cid == old_child_id else cid for cid in mol.constituents
        )
        mol.constituents = new_const
        mol.bond_energy = float(np.clip(new_coherence, 0.05, 1.0))
        if target_molecule_id not in self.parents_of.setdefault(new_child_id, []):
            self.parents_of[new_child_id].append(target_molecule_id)
        self._set_beam_strength(new_child_id, target_molecule_id, new_coherence)

        # 2) Upward propagation across inclusion hierarchy
        shifts: Dict[str, float] = {}
        queue = deque([target_molecule_id])
        visited_up: set[str] = set()
        while queue:
            cur_id = queue.popleft()
            if cur_id in visited_up:
                continue
            visited_up.add(cur_id)
            cur_node = self.nodes[cur_id]
            if len(cur_node.constituents) == 2:
                c0 = self.nodes[cur_node.constituents[0]]
                c1 = self.nodes[cur_node.constituents[1]]
                old_q = cur_node.quaternion.copy()
                new_q, new_triad = self._compute_emergent_binding(c0, c1, cur_node.bond_energy)
                cur_node.quaternion = new_q
                cur_node.chromatic_triad = new_triad
                cur_node.scale_level = max(c0.scale_level, c1.scale_level) + 1
                self._sync_node_to_causal_field(cur_node)
                shifts[cur_id] = quat_geodesic_angle(old_q, new_q)
            for p_id in self.parents_of.get(cur_id, []):
                queue.append(p_id)
        return shifts


# ======================================================================
# 3. 2-Panel Hierarchy & Chromatic Emergence Dashboard Renderer
# ======================================================================
def render_trinitarian_protocol_dashboard(
    protocol: TrinitarianScaleRotorProtocol,
    before_triads: Dict[str, np.ndarray],
    after_triads: Dict[str, np.ndarray],
    shifts: Dict[str, float],
    output_path: Path = DEFAULT_TRINITARIAN_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(wspace=0.20, top=0.88, bottom=0.12, left=0.05, right=0.97)

    # Left Panel: Trinitarian Atomic -> Molecular -> Super-Molecular Hierarchy
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    ax1.set_xlim(-0.2, 4.2)
    ax1.set_ylim(-0.3, 2.55)
    ax1.axis("off")
    ax1.set_title("Trinitarian Scale Inclusion & Zoom-In Re-Tuning", color="#e6edf3", fontsize=11)

    coords: Dict[str, Tuple[float, float]] = {
        "atom_visual_red": (0.45, 0.25),
        "atom_tactile_resist": (1.55, 0.25),
        "atom_pass_through": (2.65, 0.25),
        "atom_word_byeok": (3.65, 0.25),
        "mol_wall": (1.55, 1.15),
        "super_byeok": (2.55, 2.05),
    }

    # Draw beams first
    # 1) Active beams in CausalField
    for b in protocol.causal_field.beams:
        if b.source_id in coords and b.target_id in coords:
            x0, y0 = coords[b.source_id]
            x1, y1 = coords[b.target_id]
            if b.strength < 0.10:
                ax1.plot([x0, x1], [y0, y1], color="#ff7b72", ls="--", lw=1.6, alpha=0.85, zorder=1)
                ax1.text(0.5 * (x0 + x1) - 0.18, 0.5 * (y0 + y1), "TORN\n(0.02)", color="#ff7b72", fontsize=7.6, ha="center", zorder=2)
            else:
                ax1.plot([x0, x1], [y0, y1], color="#7ee787", lw=2.4, alpha=0.90, zorder=1)

    # 2) Node boxes
    box_colors = {
        "atom_visual_red": ("#ff7b72", "s=0 Atom\nVisual Red"),
        "atom_tactile_resist": ("#6e7681", "s=0 Atom (Detached)\nTactile Resist"),
        "atom_pass_through": ("#f2cc60", "s=0 Atom (Re-Bound)\nPass-Through"),
        "atom_word_byeok": ("#58a6ff", "s=0 Atom\nWord 'Byeok'"),
        "mol_wall": ("#d2a8ff", f"s=1 Molecule\nmol_wall (shift={shifts.get('mol_wall', 0):.2f} rad)"),
        "super_byeok": ("#7ee787", f"s=2 Super-Molecule\nsuper_byeok (shift={shifts.get('super_byeok', 0):.2f} rad)"),
    }
    for nid, (cx, cy) in coords.items():
        col, text_lbl = box_colors[nid]
        w_box = 0.88 if nid.startswith("atom") else 1.48
        h_box = 0.34
        rect = FancyBboxPatch(
            (cx - w_box / 2, cy - h_box / 2),
            w_box,
            h_box,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor="#21262d",
            edgecolor=col,
            linewidth=2.0,
            zorder=3,
        )
        ax1.add_patch(rect)
        ax1.text(cx, cy, text_lbl, ha="center", va="center", color="#e6edf3", fontsize=7.8, fontweight="bold", zorder=4)

    # Right Panel: Chromatic Emergence (Red + Blue -> Emergent Yellow & Post-ReTuning Shift)
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    bar_labels = [
        "Visual\nAtom (s=0)",
        "Resist\nAtom (s=0)",
        "Molecule\ns=1 (Before)",
        "Super-Mol\ns=2 (Before)",
        "Molecule\ns=1 (Re-Tuned)",
        "Super-Mol\ns=2 (Re-Tuned)",
    ]
    triads = [
        before_triads["atom_visual_red"],
        before_triads["atom_tactile_resist"],
        before_triads["mol_wall"],
        before_triads["super_byeok"],
        after_triads["mol_wall"],
        after_triads["super_byeok"],
    ]
    rx = np.arange(len(bar_labels))
    wb = 0.25
    reds = [float(t[0]) for t in triads]
    blues = [float(t[1]) for t in triads]
    yellows = [float(t[2]) for t in triads]

    ax2.bar(rx - wb, reds, width=wb, color="#ff7b72", label="Red (Dynamic Flux)")
    ax2.bar(rx, blues, width=wb, color="#58a6ff", label="Blue (Order / Resistance)")
    ax2.bar(rx + wb, yellows, width=wb, color="#f2cc60", label="Yellow (Emergent Character)")
    ax2.set_ylim(0.0, 1.24)
    ax2.set_xticks(rx)
    ax2.set_xticklabels(bar_labels, color="#e6edf3", fontsize=7.8)
    ax2.set_ylabel("Chromatic Triad Weight", color="#c9d1d9", fontsize=9.5)
    ax2.set_title("Emergent Chromatic Character & Upward Re-Tuning Propagation", color="#e6edf3", fontsize=11)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
