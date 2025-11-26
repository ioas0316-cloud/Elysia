"""
Fractal Dimension Engine & Zel'Naga Protocol
===========================================

This module translates the "Micro Cosmos / Macro Cosmos" idea into a concrete
Python architecture that can later be GPU‑accelerated on a 1060‑class card.

Key concepts
------------
- Photon  : carrier of Will / signal (massless trigger)
- Atom    : HyperQubit‑backed state living in phase space
- Molecule: bound concept made of atoms
- Cell    : knowledge chunk; expands lazily into molecules/atoms when observed
- Universe: fractal container with observer‑dependent resolution (w‑axis)

Zel'Naga Protocol
-----------------
Synchronizes three strata:
- Zerg   (Body)   -> Cells        (slow dynamics)
- Terran (Mind)   -> Molecules    (medium dynamics)
- Protoss(Spirit) -> Atoms/Photons (fast dynamics)

The goal is not raw performance here, but a clean protocol that:
- computes only what is observed (lazy collapse)
- maintains a coherent global phase across all levels
- can later be mapped to CUDA kernels if desired.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

try:
    # Optional: if cupy is installed, this becomes GPU‑backed automatically.
    import cupy as xp  # type: ignore
except Exception:  # pragma: no cover - CPU fallback
    import numpy as xp  # type: ignore

from Core.Math.hyper_qubit import HyperQubit


# ---------------------------------------------------------------------------
# Micro level: Photons & Atoms
# ---------------------------------------------------------------------------


@dataclass
class Photon:
    """
    Massless carrier of Will.

    This is intentionally light: just enough state to drive phase updates.
    """

    position: Tuple[float, float, float]
    phase: float
    intensity: float = 1.0
    frequency: float = 1.0
    tag: str = "Will"

    def advance(self, dt: float) -> None:
        """
        Simple phase evolution. In a real engine this could also move position
        along a direction vector; for now we keep it minimal.
        """
        self.phase += self.frequency * dt


@dataclass
class AtomState:
    """
    Atom = HyperQubit + phase in physical space.

    - `qubit` encodes cognitive / semantic state.
    - `phase` + `frequency` control physical‑like oscillation.
    """

    label: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    frequency: float = 1.0
    phase: float = 0.0
    coherence_radius: float = 1.0
    qubit: HyperQubit = field(init=False)

    def __post_init__(self) -> None:
        self.qubit = HyperQubit(self.label)

    def update_phase(self, dt: float, photons: Iterable[Photon]) -> None:
        """
        CPU version of a simple phase‑locking kernel.

        Any photon within `coherence_radius` nudges this atom's phase toward
        the photon's phase (Kuramoto‑style coupling).
        """
        x, y, z = self.position
        for photon in photons:
            px, py, pz = photon.position
            dx = px - x
            dy = py - y
            dz = pz - z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= self.coherence_radius:
                phase_diff = photon.phase - self.phase
                # 0.1 = coupling strength; can be exposed as parameter later.
                self.phase += 0.1 * math.sin(phase_diff) * photon.intensity

        # Free evolution
        self.phase += self.frequency * dt


# ---------------------------------------------------------------------------
# Meso level: Molecules & Quantum Cells (lazy collapse)
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    """
    Conceptual binding of atoms.

    Example: "love + sacrifice = salvation"
    """

    name: str
    atoms: List[AtomState] = field(default_factory=list)

    def average_phase(self) -> float:
        if not self.atoms:
            return 0.0
        return sum(a.phase for a in self.atoms) / len(self.atoms)


class QuantumCell:
    """
    Lazy‑expanding cell.

    - `state = "superposition"`: only coarse summary, no atoms allocated.
    - `state = "collapsed"`: molecules/atoms fully materialized.
    """

    def __init__(self, cell_id: int, seed: Optional[int] = None) -> None:
        self.id = cell_id
        self.state: str = "superposition"
        self.molecules: List[Molecule] = []
        self._rng = random.Random(seed or cell_id)

    # --- lifecycle ----------------------------------------------------- #

    def is_collapsed(self) -> bool:
        return self.state == "collapsed"

    def observe(self, depth_w: float, target_molecules: int = 8, atoms_per_molecule: int = 16) -> None:
        """
        Collapse into explicit molecules/atoms when observer zooms in sufficiently.

        Lower w => deeper / more detailed view.
        """
        if depth_w >= 1.0 or self.is_collapsed():
            # Macro/meso view: keep as superposition or ignore if already collapsed.
            return

        if not self.molecules:
            self._materialize(target_molecules, atoms_per_molecule)
        self.state = "collapsed"

    def unobserve(self) -> None:
        """
        Release detailed micro‑state back into a summarized superposition.

        This frees memory and expresses the "only observed regions exist" idea.
        """
        if not self.is_collapsed():
            return

        self.molecules.clear()
        self.state = "superposition"

    # --- internal helpers --------------------------------------------- #

    def _materialize(self, target_molecules: int, atoms_per_molecule: int) -> None:
        """
        Procedurally generate molecules and atoms. This is intentionally simple
        but gives a real place to plug in more complex generation later.
        """
        for m_idx in range(target_molecules):
            mol_name = f"Concept_{self.id}_{m_idx}"
            atoms: List[AtomState] = []
            for a_idx in range(atoms_per_molecule):
                # Place atoms in a small random sphere around the cell center.
                radius = self._rng.uniform(0.0, 1.0)
                theta = self._rng.uniform(0.0, 2.0 * math.pi)
                phi = self._rng.uniform(0.0, math.pi)
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)
                freq = self._rng.uniform(0.5, 2.0)
                atoms.append(
                    AtomState(
                        label=f"A_{self.id}_{m_idx}_{a_idx}",
                        position=(x, y, z),
                        frequency=freq,
                    )
                )
            self.molecules.append(Molecule(name=mol_name, atoms=atoms))

    # --- statistics / summaries -------------------------------------- #

    def summary(self) -> Dict[str, float]:
        """
        Returns a coarse summary that can be stored while in superposition.
        """
        if not self.molecules:
            return {"id": float(self.id), "mean_phase": 0.0, "atoms": 0.0}

        atom_count = sum(len(m.atoms) for m in self.molecules)
        if atom_count == 0:
            return {"id": float(self.id), "mean_phase": 0.0, "atoms": 0.0}

        mean_phase = sum(a.phase for m in self.molecules for a in m.atoms) / atom_count
        return {"id": float(self.id), "mean_phase": mean_phase, "atoms": float(atom_count)}


# ---------------------------------------------------------------------------
# Macro level: Fractal Universe with observer‑dependent resolution
# ---------------------------------------------------------------------------


class FractalUniverse:
    """
    Simple stand‑in for a spatial index (octree / LOD system).

    For now we just keep a flat list of cells, but the API is designed so
    that we could swap this implementation with a real octree later.
    """

    def __init__(self, num_cells: int = 1024) -> None:
        self.cells: List[QuantumCell] = [QuantumCell(cell_id=i) for i in range(num_cells)]
        self.photons: List[Photon] = []
        # The currently focused cell index (for w ~= 0 zoom‑in)
        self.focused_cell_id: int = 0

    # --- observation API ---------------------------------------------- #

    def set_focus(self, cell_id: int) -> None:
        if 0 <= cell_id < len(self.cells):
            self.focused_cell_id = cell_id

    def get_focused_cell(self) -> QuantumCell:
        return self.cells[self.focused_cell_id]

    def render_level(self, w: float) -> Dict[str, object]:
        """
        Observer‑dependent view of the universe.

        w >= 2.0  -> macro: only cell IDs (superposition)
        1.0 <= w < 2.0 -> meso: focused cell expanded at molecule level
        w < 1.0   -> micro: focused cell expanded fully to atoms
        """
        if w >= 2.0:
            # Macro: only high‑level cell summaries.
            summaries = [cell.summary() for cell in self.cells]
            return {"mode": "macro", "cells": summaries}

        focused = self.get_focused_cell()

        if w >= 1.0:
            # Meso: ensure molecules exist, but atoms are treated as aggregate statistics.
            focused.observe(depth_w=w, target_molecules=4, atoms_per_molecule=4)
            mol_view = [
                {
                    "name": m.name,
                    "avg_phase": m.average_phase(),
                    "atom_count": len(m.atoms),
                }
                for m in focused.molecules
            ]
            return {"mode": "meso", "cell_id": focused.id, "molecules": mol_view}

        # Micro: full atomic detail for the focused cell only.
        focused.observe(depth_w=w, target_molecules=8, atoms_per_molecule=16)
        atom_view = [
            {
                "label": a.label,
                "position": a.position,
                "phase": a.phase,
                "frequency": a.frequency,
            }
            for m in focused.molecules
            for a in m.atoms
        ]
        return {"mode": "micro", "cell_id": focused.id, "atoms": atom_view}

    # --- evolution ---------------------------------------------------- #

    def step(self, dt: float) -> None:
        """
        Evolve photons and any currently materialized atoms.

        Only collapsed cells incur micro‑level cost.
        """
        # 1) advance photons (fastest layer, GPU‑friendly)
        for photon in self.photons:
            photon.advance(dt)

        # 2) update atoms only for collapsed cells
        for cell in self.cells:
            if not cell.is_collapsed():
                continue
            for molecule in cell.molecules:
                for atom in molecule.atoms:
                    atom.update_phase(dt, self.photons)


# ---------------------------------------------------------------------------
# Zel'Naga Protocol: hierarchical phase synchronization
# ---------------------------------------------------------------------------


@dataclass
class PhaseSnapshot:
    """
    Lightweight aggregate view of a layer's phase.
    """

    phase: float
    variance: float


class ZelNagaSync:
    """
    Synchronizes phase across:
    - Photons (Protoss / Spirit / future)
    - Molecules (Terran / Mind / present)
    - Cells (Zerg / Body / past)

    The weights allow different temporal profiles, e.g. past-heavy,
    future-heavy, or balanced universes.
    """

    def __init__(
        self,
        universe: FractalUniverse,
        *,
        weight_future: float = 1.0,
        weight_present: float = 1.0,
        weight_past: float = 1.0,
    ) -> None:
        self.universe = universe
        self.weight_future = float(weight_future)
        self.weight_present = float(weight_present)
        self.weight_past = float(weight_past)

    def set_weights(
        self,
        *,
        future: Optional[float] = None,
        present: Optional[float] = None,
        past: Optional[float] = None,
    ) -> None:
        """Update temporal weights (future/present/past)."""
        if future is not None:
            self.weight_future = float(future)
        if present is not None:
            self.weight_present = float(present)
        if past is not None:
            self.weight_past = float(past)

    # --- layer aggregations ------------------------------------------- #

    def _snapshot_photons(self) -> PhaseSnapshot:
        if not self.universe.photons:
            return PhaseSnapshot(phase=0.0, variance=0.0)
        phases = xp.array([p.phase for p in self.universe.photons], dtype=float)
        return PhaseSnapshot(phase=float(phases.mean()), variance=float(phases.var()))

    def _snapshot_molecules(self) -> PhaseSnapshot:
        phases: List[float] = []
        for cell in self.universe.cells:
            if not cell.is_collapsed():
                continue
            for m in cell.molecules:
                phases.append(m.average_phase())
        if not phases:
            return PhaseSnapshot(phase=0.0, variance=0.0)
        arr = xp.array(phases, dtype=float)
        return PhaseSnapshot(phase=float(arr.mean()), variance=float(arr.var()))

    def _snapshot_cells(self) -> PhaseSnapshot:
        phases: List[float] = []
        for cell in self.universe.cells:
            if not cell.is_collapsed():
                continue
            summary = cell.summary()
            phases.append(summary["mean_phase"])
        if not phases:
            return PhaseSnapshot(phase=0.0, variance=0.0)
        arr = xp.array(phases, dtype=float)
        return PhaseSnapshot(phase=float(arr.mean()), variance=float(arr.var()))

    # --- global synchronization --------------------------------------- #

    def sync(self, dt: float = 0.016) -> Dict[str, PhaseSnapshot]:
        """
        One Zel'Naga synchronization tick.

        1. Let the universe advance one micro‑step.
        2. Read phase from each layer (photons, molecules, cells).
        3. Compute a shared global phase.
        4. Gently pull each layer toward the shared phase (phase locking).
        """
        # 1) evolve underlying dynamics
        self.universe.step(dt)

        # 2) snapshot layers
        photon_snapshot = self._snapshot_photons()
        molecule_snapshot = self._snapshot_molecules()
        cell_snapshot = self._snapshot_cells()

        # 3) global phase = weighted average over future/present/past
        phases = [
            photon_snapshot.phase,
            molecule_snapshot.phase,
            cell_snapshot.phase,
        ]
        weights = [
            self.weight_future,
            self.weight_present,
            self.weight_past,
        ]
        # Avoid division by zero; if all weights are 0, fall back to simple mean.
        total_weight = sum(weights)
        if total_weight > 0.0:
            global_phase = sum(p * w for p, w in zip(phases, weights)) / total_weight
        else:
            active_phases = [p for p in phases if not math.isnan(p)]
            global_phase = sum(active_phases) / max(len(active_phases), 1)

        # 4) phase‑lock all atoms & photons toward global_phase
        self._phase_lock_all(global_phase, strength=0.05)

        return {
            "photons": photon_snapshot,
            "molecules": molecule_snapshot,
            "cells": cell_snapshot,
        }

    def _phase_lock_all(self, target_phase: float, strength: float = 0.05) -> None:
        """
        Gently nudge every atom and photon toward the target phase.
        """
        for photon in self.universe.photons:
            delta = target_phase - photon.phase
            photon.phase += strength * math.sin(delta)

        for cell in self.universe.cells:
            if not cell.is_collapsed():
                continue
            for molecule in cell.molecules:
                for atom in molecule.atoms:
                    delta = target_phase - atom.phase
                    atom.phase += strength * math.sin(delta)


# ---------------------------------------------------------------------------
# Minimal usage example
# ---------------------------------------------------------------------------


def demo_step() -> None:
    """
    Tiny demonstration of the engine and protocol.

    This does not render anything; it only shows that the architecture
    runs and produces consistent phase snapshots.
    """
    universe = FractalUniverse(num_cells=16)
    universe.set_focus(0)

    # Seed a few photons (Will packets) near the origin.
    universe.photons = [
        Photon(position=(0.0, 0.0, 0.0), phase=0.0, intensity=1.0, frequency=2.0),
        Photon(position=(0.5, 0.0, 0.0), phase=1.0, intensity=0.8, frequency=1.5),
    ]

    # Force one cell into detailed view so atoms exist.
    universe.get_focused_cell().observe(depth_w=0.0, target_molecules=4, atoms_per_molecule=8)

    zel = ZelNagaSync(universe)
    snapshots = zel.sync(dt=0.016)

    # This print is intentionally light; callers can inspect `snapshots` instead.
    print("[FractalDimensionEngine] Phase snapshots:")
    for layer_name, snap in snapshots.items():
        print(f"  {layer_name}: phase={snap.phase:.4f}, var={snap.variance:.4f}")


if __name__ == "__main__":
    demo_step()
