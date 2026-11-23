r"""
implicit_protocol_internalizer.py

Autonomous Implicit Protocol Internalizer via Trinitarian Scale-Rotor Binding
==============================================================================
Moves beyond manual atom wiring by having ELYSIA autonomously fathom external
implicit protocols ("무엇(A)이 무엇(B) 때문에 무엇(C)으로 존재하는가") directly
from raw, unlabelled phenomena containing irrelevant ambient distractors.

Architecture:
  1. Single Universal Internalizer (`ImplicitProtocolInternalizer`) built purely on top
     of `TrinitarianScaleRotorProtocol` (`CausalField` + non-commutative $S^3$ rotors).
     Zero numeric regression (`MSE` / `lstsq`), zero per-world logic branches.
  2. Discovers why a seemingly ambiguous `L_micro` surface entity splits into distinct
     `L_meso` consequences by measuring conditional coherence over candidate `L_macro`
     context poles, explicitly rejecting random distractors.
  3. Crystallizes each discovered relational law directly into an `s=2` Trinitarian
     inclusion super-molecule `(Surface_A ⊗ Cause_B) ⊗ Outcome_C` inside `CausalField`,
     and answers unseen counterfactual queries by navigating `zoom_in_inspect`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.trinitarian_scale_rotor_protocol import (
    ScaleRotorNode,
    TrinitarianScaleRotorProtocol,
)


DEFAULT_PROTOCOL_INTERNALIZER_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_implicit_protocol_internalization_dashboard.png"
)


# ======================================================================
# 1. Domain-Agnostic Phenomenon & Discovered Protocol Data Contracts
# ======================================================================
@dataclass
class RawPhenomenonEpisode:
    """One unlabeled encounter pairing surface appearance, ambient context, and outcome."""

    surface_token: str
    context_factors: Dict[str, str]
    outcome_token: str
    outcome_chromatic: Tuple[float, float, float]


@dataclass
class DiscoveredProtocolRule:
    """Human-readable audit handle anchored to one `s=2` Trinitarian super-molecule."""

    super_molecule_id: str
    surface_atom: str
    cause_factor_key: str
    cause_atom: str
    outcome_atom: str
    ambiguity_before: float
    ambiguity_after: float
    coherence_score: float
    constituent_tree_ids: Tuple[str, ...]


@dataclass
class WorldInternalizationReport:
    """Summary of what ELYSIA fathomed vs rejected in a blind phenomenon world."""

    world_id: str
    surface_only_ambiguity: float
    protocol_residual_ambiguity: float
    discovered_cause_key: Optional[str]
    rejected_distractor_keys: Tuple[str, ...]
    discovered_rules: List[DiscoveredProtocolRule]
    unseen_prediction_accuracy: float
    factor_coherence_scores: Dict[str, float]


# ======================================================================
# 2. Three Unlabeled Blind Worlds With Irrelevant Distractor Noise
# ======================================================================
class ThermalPhaseMatterWorld:
    """
    World 1 (Ice / Water / Steam): Identical substance `matter_x` manifests as
    `RIGID_ICE`, `FLOWING_WATER`, or `DISPERSING_VAPOR` solely because of
    `ambient_thermal`, while `bg_lighting` is irrelevant random clutter.
    """

    def rollout(self, n_episodes: int = 90, seed: int = 101) -> List[RawPhenomenonEpisode]:
        rng = np.random.default_rng(seed)
        thermals = ("COLD", "WARM", "HOT")
        lights = ("DIM", "BRIGHT", "STROBE")
        mapping = {
            "COLD": ("RIGID_ICE", (0.05, 0.90, 0.05)),
            "WARM": ("FLOWING_WATER", (0.75, 0.15, 0.10)),
            "HOT": ("DISPERSING_VAPOR", (0.20, 0.05, 0.75)),
        }
        episodes: List[RawPhenomenonEpisode] = []
        for idx in range(n_episodes):
            th = thermals[idx % 3] if idx < 15 else str(rng.choice(thermals))
            lg = str(rng.choice(lights))
            out_tok, chroma = mapping[th]
            episodes.append(
                RawPhenomenonEpisode(
                    surface_token="matter_x",
                    context_factors={"ambient_thermal": th, "bg_lighting": lg},
                    outcome_token=out_tok,
                    outcome_chromatic=chroma,
                )
            )
        return episodes


class RelationalKeyGateWorld:
    """
    World 2 (Self-Tool Relational Affordance): Same barrier `barrier_g` behaves as
    `BLOCKED_WALL` vs `OPEN_PASSAGE` strictly due to ELYSIA's `self_coupling` pole,
    while `sky_tint` is pure random noise.
    """

    def rollout(self, n_episodes: int = 80, seed: int = 202) -> List[RawPhenomenonEpisode]:
        rng = np.random.default_rng(seed)
        couplings = ("EMPTY_HAND", "HOLDING_TOKEN")
        skies = ("RED", "BLUE", "GRAY")
        mapping = {
            "EMPTY_HAND": ("BLOCKED_WALL", (0.05, 0.90, 0.05)),
            "HOLDING_TOKEN": ("OPEN_PASSAGE", (0.45, 0.05, 0.50)),
        }
        episodes: List[RawPhenomenonEpisode] = []
        for idx in range(n_episodes):
            cp = couplings[idx % 2] if idx < 10 else str(rng.choice(couplings))
            sk = str(rng.choice(skies))
            out_tok, chroma = mapping[cp]
            episodes.append(
                RawPhenomenonEpisode(
                    surface_token="barrier_g",
                    context_factors={"self_coupling": cp, "sky_tint": sk},
                    outcome_token=out_tok,
                    outcome_chromatic=chroma,
                )
            )
        return episodes


class BoundaryClosureSyntaxWorld:
    """
    World 3 (Structural / Execution Protocol): Identical payload `expr_p` passes as
    `EXEC_ACCEPTED` only when `outer_boundary == BALANCED_PAIR` and fails otherwise,
    while `whitespace_pad` is pure noise.
    """

    def rollout(self, n_episodes: int = 80, seed: int = 303) -> List[RawPhenomenonEpisode]:
        rng = np.random.default_rng(seed)
        boundaries = ("BALANCED_PAIR", "BALANCED_PAIR", "OPEN_ONLY", "CLOSE_ONLY")
        pads = ("NONE", "WIDE")
        mapping = {
            "BALANCED_PAIR": ("EXEC_ACCEPTED", (0.15, 0.70, 0.15)),
            "OPEN_ONLY": ("EXEC_REJECTED", (0.80, 0.10, 0.10)),
            "CLOSE_ONLY": ("EXEC_REJECTED", (0.80, 0.10, 0.10)),
        }
        episodes: List[RawPhenomenonEpisode] = []
        for idx in range(n_episodes):
            bd = boundaries[idx % 4] if idx < 16 else str(rng.choice(boundaries))
            pd = str(rng.choice(pads))
            out_tok, chroma = mapping[bd]
            episodes.append(
                RawPhenomenonEpisode(
                    surface_token="expr_p",
                    context_factors={"outer_boundary": bd, "whitespace_pad": pd},
                    outcome_token=out_tok,
                    outcome_chromatic=chroma,
                )
            )
        return episodes


# ======================================================================
# 3. Unified Implicit Protocol Internalizer
# ======================================================================
class ImplicitProtocolInternalizer:
    """
    Single domain-agnostic architecture resting on `TrinitarianScaleRotorProtocol`.
    Discovers the causal boundary factor explaining why one surface entity produces
    different consequences, crystallizes `(Surface ⊗ Cause) ⊗ Outcome` super-molecules
    in `CausalField`, and reasons purely over its own inclusion tree on unseen episodes.
    """

    def __init__(self) -> None:
        self.protocol = TrinitarianScaleRotorProtocol()
        self.world_super_molecules: Dict[str, List[str]] = {}
        self.reports: List[WorldInternalizationReport] = []

    @staticmethod
    def _deterministic_axis_angle(token_key: str) -> Tuple[Tuple[float, float, float], float]:
        digest = hashlib.sha256(token_key.encode("utf-8")).digest()
        raw = [float(b) / 255.0 - 0.5 for b in digest[:4]]
        ax = np.array(raw[:3], dtype=np.float64)
        norm = float(np.linalg.norm(ax))
        if norm < 1e-6:
            ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            ax = ax / norm
        angle = 0.55 + 1.15 * (float(digest[3]) / 255.0)
        return (float(ax[0]), float(ax[1]), float(ax[2])), float(angle)

    def _ensure_atom(
        self,
        atom_id: str,
        chromatic_triad: Tuple[float, float, float],
    ) -> ScaleRotorNode:
        if atom_id in self.protocol.nodes:
            return self.protocol.nodes[atom_id]
        axis, angle = self._deterministic_axis_angle(atom_id)
        return self.protocol.register_atom(atom_id, axis, angle, chromatic_triad)

    @staticmethod
    def _compute_surface_ambiguity(episodes: List[RawPhenomenonEpisode]) -> float:
        if not episodes:
            return 0.0
        by_surface: Dict[str, Dict[str, int]] = {}
        for ep in episodes:
            counts = by_surface.setdefault(ep.surface_token, {})
            counts[ep.outcome_token] = counts.get(ep.outcome_token, 0) + 1
        correct = sum(max(c.values()) for c in by_surface.values())
        return float(1.0 - (correct / len(episodes)))

    @staticmethod
    def _evaluate_factor_coherence(
        episodes: List[RawPhenomenonEpisode], factor_key: str
    ) -> Tuple[float, Dict[Tuple[str, str], Tuple[str, Tuple[float, float, float], float]]]:
        """
        Measures how cleanly knowing `(surface_token, factor_key=val)` resolves outcome ambiguity.
        Returns `(overall_coherence, pair_rules[(surface, val) -> (dom_outcome, mean_chroma, purity)])`.
        """
        grouped: Dict[Tuple[str, str], List[RawPhenomenonEpisode]] = {}
        for ep in episodes:
            val = ep.context_factors[factor_key]
            grouped.setdefault((ep.surface_token, val), []).append(ep)

        total_correct = 0
        pair_rules: Dict[Tuple[str, str], Tuple[str, Tuple[float, float, float], float]] = {}
        for pair, grp in grouped.items():
            counts: Dict[str, int] = {}
            for ep in grp:
                counts[ep.outcome_token] = counts.get(ep.outcome_token, 0) + 1
            dom_outcome = max(counts.items(), key=lambda kv: kv[1])[0]
            dom_eps = [ep for ep in grp if ep.outcome_token == dom_outcome]
            mean_chroma_arr = np.mean([ep.outcome_chromatic for ep in dom_eps], axis=0)
            mean_chroma = (
                float(mean_chroma_arr[0]),
                float(mean_chroma_arr[1]),
                float(mean_chroma_arr[2]),
            )
            purity = float(counts[dom_outcome] / len(grp))
            total_correct += counts[dom_outcome]
            pair_rules[pair] = (dom_outcome, mean_chroma, purity)

        overall_coherence = float(total_correct / max(len(episodes), 1))
        return overall_coherence, pair_rules

    def predict_via_inclusion_tree(
        self, world_id: str, episode: RawPhenomenonEpisode
    ) -> Optional[str]:
        """
        Predicts consequence of an unseen episode purely by zooming into this world's
        internalized `s=2` Trinitarian super-molecules and checking which molecular
        inclusion tree contains BOTH the observed surface atom AND matching context atom.
        """
        surf_atom = f"{world_id}::surf::{episode.surface_token}"
        candidate_causes = {
            f"{world_id}::cause::{fk}={fv}" for fk, fv in episode.context_factors.items()
        }
        out_prefix = f"{world_id}::out::"

        best_pred: Optional[str] = None
        best_energy = -1.0
        for super_id in self.world_super_molecules.get(world_id, []):
            tree_nodes = self.protocol.zoom_in_inspect(super_id)
            tree_ids = {n.node_id for n in tree_nodes}
            if surf_atom in tree_ids and any(c in tree_ids for c in candidate_causes):
                super_node = self.protocol.nodes[super_id]
                # Super-molecule binds (mol_cond, outcome_atom) -> read outcome sibling directly
                for child_id in super_node.constituents:
                    if child_id.startswith(out_prefix):
                        if super_node.bond_energy > best_energy:
                            best_energy = super_node.bond_energy
                            best_pred = child_id[len(out_prefix):]
        return best_pred

    def internalize_world(
        self,
        world_id: str,
        train_episodes: List[RawPhenomenonEpisode],
        unseen_episodes: List[RawPhenomenonEpisode],
    ) -> WorldInternalizationReport:
        """
        Single unified confrontation method across ANY domain (zero world-name checks):
        measures surface-only ambiguity, audits all candidate L_macro factors, rejects
        distractors, crystallizes causal `(A ⊗ B) ⊗ C` super-molecules, and verifies on unseen data.
        """
        surf_ambiguity = self._compute_surface_ambiguity(train_episodes)
        factor_keys = sorted(train_episodes[0].context_factors.keys()) if train_episodes else []

        factor_coherences: Dict[str, float] = {}
        factor_details: Dict[str, Dict[Tuple[str, str], Tuple[str, Tuple[float, float, float], float]]] = {}
        for fk in factor_keys:
            coh, details = self._evaluate_factor_coherence(train_episodes, fk)
            factor_coherences[fk] = coh
            factor_details[fk] = details

        best_key: Optional[str] = None
        best_coh = -1.0
        for fk in factor_keys:
            coh = factor_coherences[fk]
            res_amb = 1.0 - coh
            amb_drop = surf_ambiguity - res_amb
            if amb_drop >= 0.30 and coh >= 0.85 and coh > best_coh:
                best_coh = coh
                best_key = fk

        rejected = tuple(
            fk
            for fk in factor_keys
            if fk != best_key and (surf_ambiguity - (1.0 - factor_coherences[fk])) < 0.15
        )

        rules: List[DiscoveredProtocolRule] = []
        self.world_super_molecules[world_id] = []

        if best_key is not None:
            for (surf_tok, cause_val), (dom_out, out_chroma, purity) in sorted(
                factor_details[best_key].items()
            ):
                surf_id = f"{world_id}::surf::{surf_tok}"
                cause_id = f"{world_id}::cause::{best_key}={cause_val}"
                out_id = f"{world_id}::out::{dom_out}"

                # Register s=0 atoms organically from encountered experience
                self._ensure_atom(surf_id, (0.60, 0.25, 0.15))
                self._ensure_atom(cause_id, (0.15, 0.70, 0.15))
                self._ensure_atom(out_id, out_chroma)

                # Step 1 binding: s=0 (Surface A) ⊗ s=0 (Cause B) -> s=1 Condition Molecule
                cond_mol = self.protocol.resonate_and_bind(
                    surf_id,
                    cause_id,
                    bond_coherence=purity,
                    molecule_id=f"{world_id}::cond::{surf_tok}__{cause_val}",
                )
                # Step 2 binding: s=1 (Condition A⊗B) ⊗ s=0 (Outcome C) -> s=2 Protocol Super-Molecule
                super_mol = self.protocol.resonate_and_bind(
                    cond_mol.node_id,
                    out_id,
                    bond_coherence=purity,
                    molecule_id=f"{world_id}::proto::{surf_tok}__{cause_val}__{dom_out}",
                )
                self.world_super_molecules[world_id].append(super_mol.node_id)

                tree_nodes = self.protocol.zoom_in_inspect(super_mol.node_id)
                rules.append(
                    DiscoveredProtocolRule(
                        super_molecule_id=super_mol.node_id,
                        surface_atom=surf_id,
                        cause_factor_key=best_key,
                        cause_atom=cause_id,
                        outcome_atom=out_id,
                        ambiguity_before=surf_ambiguity,
                        ambiguity_after=1.0 - best_coh,
                        coherence_score=purity,
                        constituent_tree_ids=tuple(n.node_id for n in tree_nodes),
                    )
                )

        # Evaluate internalized inclusion tree on unseen counterfactual encounters
        unseen_correct = 0
        for u_ep in unseen_episodes:
            pred = self.predict_via_inclusion_tree(world_id, u_ep)
            if pred == u_ep.outcome_token:
                unseen_correct += 1
        unseen_acc = float(unseen_correct / max(len(unseen_episodes), 1))

        rep = WorldInternalizationReport(
            world_id=world_id,
            surface_only_ambiguity=surf_ambiguity,
            protocol_residual_ambiguity=(1.0 - best_coh) if best_key is not None else surf_ambiguity,
            discovered_cause_key=best_key,
            rejected_distractor_keys=rejected,
            discovered_rules=rules,
            unseen_prediction_accuracy=unseen_acc,
            factor_coherence_scores=factor_coherences,
        )
        self.reports.append(rep)
        return rep


# ======================================================================
# 4. 4-Panel Dashboard Renderer
# ======================================================================
def render_implicit_protocol_dashboard(
    internalizer: ImplicitProtocolInternalizer,
    reports: List[WorldInternalizationReport],
    output_path: Path = DEFAULT_PROTOCOL_INTERNALIZER_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.35, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.97)

    short_names = [f"W{i+1}: {r.world_id.split('_', 1)[-1]}" for i, r in enumerate(reports)]
    x_idx = np.arange(len(reports))
    wb = 0.35

    # Panel 1: Surface Ambiguity vs Post-Internalization Residual Ambiguity
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    amb_pre = [r.surface_only_ambiguity for r in reports]
    amb_post = [r.protocol_residual_ambiguity for r in reports]
    ax1.bar(x_idx - wb / 2, amb_pre, width=wb, color="#ff7b72", alpha=0.88, label="Surface Phenomenon Only (L_micro)")
    ax1.bar(x_idx + wb / 2, amb_post, width=wb, color="#7ee787", alpha=0.92, label="After Internalized Protocol ((A ⊗ B) ⊗ C)")
    for i in range(len(reports)):
        ax1.text(i - wb / 2, amb_pre[i] + 0.02, f"{amb_pre[i]:.2f}", ha="center", color="#ff7b72", fontsize=8, fontweight="bold")
        ax1.text(i + wb / 2, amb_post[i] + 0.02, f"{amb_post[i]:.2f}", ha="center", color="#7ee787", fontsize=8, fontweight="bold")
    ax1.set_ylim(0.0, 0.92)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(short_names, color="#e6edf3", fontsize=8.2)
    ax1.set_ylabel("Outcome Ambiguity (0 = Deterministic)", color="#c9d1d9", fontsize=9)
    ax1.set_title("Resolving Surface Ambiguity via Implicit Protocol Discovery", color="#e6edf3", fontsize=10.5)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.6)

    # Panel 2: True Causal Factor vs Rejected Distractor Coherence
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    true_cohs = [r.factor_coherence_scores[r.discovered_cause_key] for r in reports]
    dist_cohs = [r.factor_coherence_scores[r.rejected_distractor_keys[0]] for r in reports]
    ax2.bar(x_idx - wb / 2, dist_cohs, width=wb, color="#8b949e", alpha=0.85, label="Random Distractor Factor (Rejected)")
    ax2.bar(x_idx + wb / 2, true_cohs, width=wb, color="#58a6ff", alpha=0.92, label="Discovered True Cause Factor")
    for i, r in enumerate(reports):
        dist_lbl = r.rejected_distractor_keys[0].replace("_", "\n")
        cause_lbl = r.discovered_cause_key.replace("_", "\n")
        ax2.text(i - wb / 2, dist_cohs[i] + 0.02, f"{dist_lbl}\n({dist_cohs[i]:.2f})", ha="center", color="#8b949e", fontsize=7.0)
        ax2.text(i + wb / 2, true_cohs[i] + 0.02, f"{cause_lbl}\n({true_cohs[i]:.2f})", ha="center", color="#58a6ff", fontsize=7.0, fontweight="bold")
    ax2.set_ylim(0.0, 1.44)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(short_names, color="#e6edf3", fontsize=8.2)
    ax2.set_ylabel("Conditional Factor Coherence", color="#c9d1d9", fontsize=9)
    ax2.set_title("Autonomous Cause Fathoming vs Noise Rejection", color="#e6edf3", fontsize=10.5)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper center", ncol=2, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2)

    # Panel 3: Unseen Accuracy & Crystallized Super-Molecules in CausalField
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    accs = [r.unseen_prediction_accuracy * 100.0 for r in reports]
    n_rules = [len(r.discovered_rules) for r in reports]
    ax3.bar(x_idx, accs, width=0.45, color="#d2a8ff", alpha=0.88, label="Unseen Query Accuracy via zoom_in_inspect (%)")
    for i in range(len(reports)):
        ax3.text(i, accs[i] + 2.5, f"{accs[i]:.1f}%\n({n_rules[i]} s=2 mols)", ha="center", color="#e6edf3", fontsize=8.0, fontweight="bold")
    ax3.set_ylim(0.0, 134.0)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(short_names, color="#e6edf3", fontsize=8.2)
    ax3.set_ylabel("Unseen Accuracy (%)", color="#c9d1d9", fontsize=9)
    ax3.set_title("Zero-Regression Unseen Inference Over Inclusion Tree", color="#e6edf3", fontsize=10.5)
    ax3.tick_params(colors="#c9d1d9", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 4: Structural Protocol Self-Explanation Table
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    ax4.set_title("Fathomed Trinitarian Protocols Inside Single CausalField", color="#e6edf3", fontsize=10.5)
    y_pos = 0.92
    for i, r in enumerate(reports):
        ax4.text(
            0.03,
            y_pos,
            f"[{r.world_id}] Cause: {r.discovered_cause_key} | Rejected: {', '.join(r.rejected_distractor_keys)}",
            color="#58a6ff",
            fontsize=8.3,
            fontweight="bold",
            transform=ax4.transAxes,
        )
        y_pos -= 0.075
        for rule in r.discovered_rules:
            surf_short = rule.surface_atom.split("::")[-1]
            cause_short = rule.cause_atom.split("::")[-1]
            out_short = rule.outcome_atom.split("::")[-1]
            ax4.text(
                0.06,
                y_pos,
                f"({surf_short}  ⊗  {cause_short})  ⊗  {out_short}   [s=2, coh={rule.coherence_score:.2f}]",
                color="#7ee787",
                fontsize=7.9,
                family="monospace",
                transform=ax4.transAxes,
            )
            y_pos -= 0.068
        y_pos -= 0.035

    total_nodes = len(internalizer.protocol.nodes)
    total_beams = len(internalizer.protocol.causal_field.beams)
    ax4.text(
        0.03,
        0.03,
        f"Shared Persistent CausalField -> {total_nodes} ScaleRotorNodes | {total_beams} ConnectivityBeams",
        color="#ffa657",
        fontsize=8.3,
        fontweight="bold",
        transform=ax4.transAxes,
    )

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
