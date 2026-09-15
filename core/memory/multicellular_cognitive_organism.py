"""
Multicellular Cognitive Organism & Lorentzian Resonance Medium for Elysia.

This module implements:
1. TensionWave: Structural tension wave propagating through topological manifold medium.
2. CognitiveCell & ModularCognitiveCell: Topological cognitive cells with natural resonance
   frequencies w0, Q-factors, position coordinates, and attached biological boundary modules.
3. TopologicalManifoldMedium: Medium handling zero-parsing-overhead Lorentzian resonance
   frequency filtering R(w) = 1 / sqrt(1 + Q^2 ((w - w0)/w0)^2), tension osmosis, distance decay,
   LTP scar weight metric channel deformation, and adaptive cell mitosis when friction capacity is exceeded.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple

from core.memory.semantic_valence_manifold import (
    BoundaryModule,
    FieldFluxSignal,
    LTPScarringEngine
)


@dataclass
class TensionWave:
    """Tension wave propagating through the topological manifold medium."""
    source_id: str
    frequency: float       # Wave resonant frequency (Hz)
    amplitude: float       # Wave amplitude (Energy / Friction)
    origin_pos: np.ndarray # Origin 3D position [x, y, z]


class CognitiveCell:
    """
    Topological Cognitive Cell.
    Resonates with incoming tension waves based on Lorentzian response curve R(w).
    No discrete parsing overhead: energy absorption directly relaxes internal state.
    """

    def __init__(
        self,
        cell_id: str,
        natural_frequency: float,
        position: np.ndarray,
        quality_factor: float = 10.0,
        friction_capacity: float = 10.0
    ):
        self.cell_id = cell_id
        self.natural_frequency = natural_frequency # w0
        self.position = position.copy()
        self.quality_factor = quality_factor       # Q-factor
        self.friction_capacity = friction_capacity # Max friction before mitosis

        self.tension: float = 0.1
        self.accumulated_friction: float = 0.0
        self.generation: int = 1

    def calculate_resonance_factor(self, wave_freq: float) -> float:
        """
        Calculates Lorentzian resonance factor:
        R(w) = 1 / sqrt(1 + Q^2 * ((w - w0) / w0)^2)
        """
        if self.natural_frequency <= 0.0:
            return 0.0
        delta_w = (wave_freq - self.natural_frequency) / self.natural_frequency
        return float(1.0 / np.sqrt(1.0 + (self.quality_factor * delta_w) ** 2))

    def receive_wave(self, wave: TensionWave, decay_alpha: float = 0.4) -> float:
        """Absorbs wave resonance energy without string/data parsing."""
        if wave.source_id == self.cell_id:
            return 0.0

        distance = np.linalg.norm(self.position - wave.origin_pos)
        decayed_amp = wave.amplitude * np.exp(-decay_alpha * distance)
        res_factor = self.calculate_resonance_factor(wave.frequency)

        absorbed_energy = decayed_amp * res_factor
        self.tension += absorbed_energy * 0.5
        self.accumulated_friction += absorbed_energy * 0.2

        return absorbed_energy

    def emit_wave(self, friction_energy: float) -> TensionWave:
        """Emits a tension wave at cell's natural resonant frequency upon local friction."""
        self.tension += friction_energy * 0.1
        self.accumulated_friction += friction_energy * 0.5
        return TensionWave(
            source_id=self.cell_id,
            frequency=self.natural_frequency,
            amplitude=friction_energy,
            origin_pos=self.position.copy()
        )

    def check_and_mitosis(self) -> Optional['CognitiveCell']:
        """
        Checks if accumulated friction exceeds capacity.
        If exceeded, splits into a child cognitive cell along a specialized frequency axis.
        """
        if self.accumulated_friction >= self.friction_capacity:
            child_id = f"{self.cell_id}_DAUGHTER_{self.generation}"
            # Daughter cell inherits shifted frequency and offset position
            daughter_freq = self.natural_frequency * 1.05
            daughter_pos = self.position + np.array([0.2, 0.2, 0.0])

            daughter = CognitiveCell(
                cell_id=child_id,
                natural_frequency=daughter_freq,
                position=daughter_pos,
                quality_factor=self.quality_factor,
                friction_capacity=self.friction_capacity
            )
            daughter.generation = self.generation + 1
            self.generation += 1
            self.accumulated_friction *= 0.3 # Reset/relieve friction after mitosis
            return daughter

        return None


class ModularCognitiveCell(CognitiveCell):
    """Cognitive cell supporting attached biological boundary condition modules at ∂Ω."""

    def __init__(
        self,
        cell_id: str,
        natural_frequency: float,
        position: np.ndarray,
        quality_factor: float = 10.0,
        friction_capacity: float = 10.0
    ):
        super().__init__(cell_id, natural_frequency, position, quality_factor, friction_capacity)
        self.boundary_modules: Dict[str, BoundaryModule] = {}

    def attach_boundary_module(self, name: str, module: BoundaryModule):
        self.boundary_modules[name] = module

    def detach_boundary_module(self, name: str):
        if name in self.boundary_modules:
            del self.boundary_modules[name]

    def receive_wave_modular(self, wave: TensionWave, current_time: float, decay_alpha: float = 0.4) -> float:
        """Processes wave flux through attached boundary module pipeline."""
        if wave.source_id == self.cell_id:
            return 0.0

        distance = np.linalg.norm(self.position - wave.origin_pos)
        decayed_amp = wave.amplitude * np.exp(-decay_alpha * distance)
        res_factor = self.calculate_resonance_factor(wave.frequency)
        raw_absorbed = decayed_amp * res_factor

        # Construct flux signal
        flux_signal = FieldFluxSignal(
            amplitude=raw_absorbed,
            frequency=wave.frequency,
            gradient=np.array([raw_absorbed, -raw_absorbed, 0.0])
        )

        # Pass through attached boundary modules
        processed_signal = flux_signal
        for mod in self.boundary_modules.values():
            processed_signal = mod.apply_boundary_condition(processed_signal, current_time)

        final_absorbed = processed_signal.amplitude
        self.tension += final_absorbed * 0.5
        self.accumulated_friction += final_absorbed * 0.2

        return final_absorbed


class TopologicalManifoldMedium:
    """
    Topological Manifold Medium propagating tension waves across cognitive cells.
    Handles global resonance synchronization, tension osmosis, and dynamic cell mitosis.
    """

    def __init__(self, decay_alpha: float = 0.4):
        self.cells: Dict[str, CognitiveCell] = {}
        self.decay_alpha = decay_alpha
        self.ltp_engine = LTPScarringEngine()

    def register_cell(self, cell: CognitiveCell):
        self.cells[cell.cell_id] = cell

    def propagate_wave_event(
        self,
        source_id: str,
        friction_energy: float,
        current_time: float = 0.0
    ) -> List[Tuple[str, float, float]]:
        """
        1. Emits tension wave from source cell.
        2. Propagates wave through medium to all registered cells via Lorentzian resonance.
        3. Integrates LTP scarring for repeated resonance.
        4. Triggers cell mitosis if friction capacity is exceeded.
        """
        if source_id not in self.cells:
            return []

        source_cell = self.cells[source_id]
        wave = source_cell.emit_wave(friction_energy)

        # Apply LTP scarring integration
        flux = FieldFluxSignal(
            amplitude=friction_energy,
            frequency=wave.frequency,
            gradient=np.array([1.0, 1.0, 1.0])
        )
        self.ltp_engine.apply_boundary_condition(flux, current_time)

        results = []
        new_cells = []

        for cell_id, cell in list(self.cells.items()):
            if cell_id == source_id:
                continue

            if isinstance(cell, ModularCognitiveCell):
                absorbed = cell.receive_wave_modular(wave, current_time, decay_alpha=self.decay_alpha)
            else:
                absorbed = cell.receive_wave(wave, decay_alpha=self.decay_alpha)

            res_factor = cell.calculate_resonance_factor(wave.frequency)
            results.append((cell_id, res_factor, absorbed))

            # Check for mitosis
            daughter = cell.check_and_mitosis()
            if daughter:
                new_cells.append(daughter)

        # Register newly split cells
        for daughter in new_cells:
            self.register_cell(daughter)

        return results
