"""Hierarchical, reversible motion field for scale-coupled world objects.

This prototype stores a world's *generating rules* (rotors and parent links),
then derives positions at a requested time.  It is intended for macro motion
such as orbits, rotation, and slowly changing illumination, not pixel rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


Vector = Tuple[float, float, float]


def _unit(vector: Vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("rotor axis must have non-zero length")
    return value / norm


@dataclass(frozen=True)
class ScaleRotor:
    """A continuous generator: axis, angular rate, phase, radius, and scale."""

    axis: Vector = (0.0, 0.0, 1.0)
    angular_rate: float = 1.0
    phase: float = 0.0
    radius: float = 1.0
    scale: float = 1.0

    def angle_at(self, time: float) -> float:
        return self.phase + self.angular_rate * time

    def offset_at(self, time: float) -> np.ndarray:
        """Position in the rotor plane at arbitrary time, including rewind."""
        axis = _unit(self.axis)
        # Pick a stable in-plane basis; the second vector completes the plane.
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, reference)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angle = self.angle_at(time)
        return self.radius * self.scale * (np.cos(angle) * u + np.sin(angle) * v)


@dataclass
class RotorNode:
    """A causal object whose local motion is defined relative to a parent."""

    node_id: str
    rotor: ScaleRotor
    parent_id: Optional[str] = None
    local_origin: Vector = (0.0, 0.0, 0.0)
    # Optional spectral bands represent motion at different temporal scales.
    bands: Tuple[ScaleRotor, ...] = field(default_factory=tuple)


class CausalScaleRotorField:
    """Evaluate nested rotors as a worldline, forward or backward in time.

    Nodes retain their causal parent links and motion generators.  Querying a
    time slice composes only the requested node's ancestry; no frame history or
    full scene tensor is required.  `advance` uses exact analytic phase deltas
    for any requested time and returns changed positions for consumers.
    """

    def __init__(self, nodes: Iterable[RotorNode] = (), causal_field=None) -> None:
        self.nodes: Dict[str, RotorNode] = {}
        self._last_positions: Dict[str, np.ndarray] = {}
        self._time: Optional[float] = None
        self.causal_field = causal_field
        for node in nodes:
            self.add(node)

    def add(self, node: RotorNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate rotor node: {node.node_id}")
        if node.parent_id is not None and node.parent_id not in self.nodes:
            raise ValueError(f"parent must be registered first: {node.parent_id}")
        self.nodes[node.node_id] = node
        if self.causal_field is not None:
            # Bind the generator to Elysia's shared causal substrate.  The
            # rotor remains the authoritative rule; the voxel carries its
            # current observable state and causal signature.
            from core.physics.causal_field import InformationVoxel

            self.causal_field.add_voxel(
                InformationVoxel(
                    id=node.node_id,
                    content={"rotor": node.rotor, "parent_id": node.parent_id},
                    tensor=np.asarray(
                        [node.rotor.angular_rate, node.rotor.phase, node.rotor.radius, node.rotor.scale],
                        dtype=np.float32,
                    ),
                    mass=max(abs(node.rotor.scale), 1e-6),
                    position=np.asarray(node.local_origin, dtype=np.float32),
                    velocity=np.zeros(3, dtype=np.float32),
                    phase_angle=node.rotor.phase,
                    scale_level=abs(node.rotor.scale),
                )
            )

    def position_at(self, node_id: str, time: float) -> np.ndarray:
        """Resolve a causal world position at any time, including the past."""
        node = self.nodes[node_id]
        local = np.asarray(node.local_origin, dtype=np.float64) + node.rotor.offset_at(time)
        for band in node.bands:
            local += band.offset_at(time)
        if node.parent_id is None:
            return local
        return self.position_at(node.parent_id, time) + local

    def slice_at(self, time: float, node_ids: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
        """Extract a spatial cross-section of the worldline at arbitrary time."""
        selected = self.nodes.keys() if node_ids is None else node_ids
        return {node_id: self.position_at(node_id, time) for node_id in selected}

    def advance(self, time: float) -> Dict[str, np.ndarray]:
        """Return per-node displacement from the previous slice (causal delta)."""
        current = self.slice_at(time)
        previous_positions = self._last_positions
        previous_time = self._time
        if self._time is None:
            changes = {key: value.copy() for key, value in current.items()}
        else:
            changes = {
                key: value - self._last_positions[key]
                for key, value in current.items()
                if not np.allclose(value, self._last_positions[key], rtol=0.0, atol=1e-12)
            }
        if self.causal_field is not None:
            for node_id, position in current.items():
                voxel = self.causal_field.voxels[node_id]
                voxel.position = position.astype(np.float32)
                if node_id in previous_positions and previous_time is not None:
                    voxel.velocity = (
                        (position - previous_positions[node_id])
                        / ((time - previous_time) if time != previous_time else 1e-12)
                    ).astype(np.float32)
                voxel.phase_angle = self.nodes[node_id].rotor.angle_at(time) % (2.0 * np.pi)
        self._last_positions = current
        self._time = time
        return changes


def sample_solar_system(time: float) -> Dict[str, np.ndarray]:
    """Small demonstration: star, planet orbit, moon orbit, and axial spin."""
    field = CausalScaleRotorField(
        [
            RotorNode("star", ScaleRotor(radius=0.0)),
            RotorNode("planet", ScaleRotor(angular_rate=0.2, radius=10.0), "star"),
            RotorNode(
                "moon",
                ScaleRotor(angular_rate=1.4, radius=2.0),
                "planet",
                bands=(ScaleRotor(angular_rate=0.8, radius=0.25),),
            ),
        ]
    )
    return field.slice_at(time)
