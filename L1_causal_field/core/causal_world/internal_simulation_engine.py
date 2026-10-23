r"""
Internal Simulation Engine (내부 월드 시뮬레이션 엔진 및 지속적 틱 루프)
========================================================================

Implements the internal simulation sandbox, continuous breathing tick loop (while(1) / tick(dt)),
virtual physics collision detection, tension wave propagation, and digital twin state synchronization.

Key Components:
1. VirtualPhysicsCollisionEngine:
   Detects collisions & interrupts between internal concept nodes/voxels and incoming raw streams.
   Computes Tension Wave propagation: $\nabla \cdot \mathbf{T} + \mathbf{F}_{collision} = m \mathbf{a}$.
2. ContinuousTickLoop:
   Background thread and explicit tick(dt) loop that continuously updates internal state,
   oscillators, friction hysteresis, and void gradients without waiting for user prompt.
3. DigitalTwinSyncInterface:
   Translates internal void gradients ($\nabla V_{void}$) and Back-EMF resistance ($E_{back}$)
   into active seeking vectors and digital twin external registers.
4. InternalSimulationEngine:
   Integrates virtual collision engine, continuous tick loop, and digital twin interface
   into a unified internal world simulation sandbox.
"""

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np


@dataclass
class ConceptNode:
    """A node / voxel inside the internal simulation sandbox."""
    node_id: str
    position: np.ndarray  # Position vector in continuous manifold space
    velocity: np.ndarray  # Velocity vector
    mass: float = 1.0
    tension: float = 0.0
    phase: float = 0.0


@dataclass
class CollisionEvent:
    """Occurs when an incoming external stream or internal concept collides with a node."""
    node_id: str
    impact_force: np.ndarray
    tension_wave_magnitude: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SimulationStateReport:
    """Snapshot report of the internal simulation engine state."""
    tick_count: int
    dt: float
    total_energy: float
    accumulated_tension: float
    active_collisions_count: int
    nodes_positions: Dict[str, np.ndarray]
    void_gradient_magnitude: float


class VirtualPhysicsCollisionEngine:
    """
    Virtual Physics & Collision Engine.
    Detects topological collisions and boundary interrupts when new streams
    or internal nodes collide. Propagates tension waves across the internal manifold.
    """
    def __init__(self, dim: int = 2, boundary_limit: float = 5.0):
        self.dim = dim
        self.boundary_limit = boundary_limit

    def detect_and_resolve_collisions(
        self,
        nodes: Dict[str, ConceptNode],
        external_wave: Optional[np.ndarray] = None,
        dt: float = 0.05,
    ) -> List[CollisionEvent]:
        events = []
        node_ids = list(nodes.keys())

        # 1. Node-to-Node Collisions
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                n1 = nodes[node_ids[i]]
                n2 = nodes[node_ids[j]]
                dist_vec = n1.position - n2.position
                dist = float(np.linalg.norm(dist_vec)) + 1e-8

                # Collision threshold radius = 1.0
                if dist < 1.0:
                    overlap = 1.0 - dist
                    normal = dist_vec / dist
                    impact_force = normal * overlap * 10.0

                    # Apply forces
                    n1.velocity += (impact_force / n1.mass) * dt
                    n2.velocity -= (impact_force / n2.mass) * dt

                    n1.tension += float(overlap)
                    n2.tension += float(overlap)

                    events.append(
                        CollisionEvent(
                            node_id=n1.node_id,
                            impact_force=impact_force,
                            tension_wave_magnitude=float(overlap),
                        )
                    )

        # 2. External Wave Collision
        if external_wave is not None:
            wave_force = external_wave[: self.dim] if len(external_wave) >= self.dim else np.pad(external_wave, (0, self.dim - len(external_wave)))
            wave_mag = float(np.linalg.norm(wave_force))

            if wave_mag > 0.1:
                for node in nodes.values():
                    node.velocity += (wave_force / node.mass) * dt
                    node.tension += wave_mag * 0.5
                    events.append(
                        CollisionEvent(
                            node_id=node.node_id,
                            impact_force=wave_force,
                            tension_wave_magnitude=wave_mag,
                        )
                    )

        # 3. Boundary Collisions
        for node in nodes.values():
            for d in range(self.dim):
                if abs(node.position[d]) > self.boundary_limit:
                    overflow = abs(node.position[d]) - self.boundary_limit
                    bounce_force = -np.sign(node.position[d]) * overflow * 20.0
                    node.velocity[d] += (bounce_force / node.mass) * dt
                    node.tension += overflow

        return events


class DigitalTwinSyncInterface:
    """
    Digital Twin Synchronization Interface.
    Translates internal void gradients ($\nabla V_{void}$), Back-EMF resistance ($E_{back}$),
    and simulation state into real-world action vectors and external registers.
    """
    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim
        self.external_registers: Dict[str, Any] = {}

    def sync_to_world(
        self,
        void_gradient: float,
        back_emf: float,
        nodes: Dict[str, ConceptNode],
    ) -> np.ndarray:
        """
        Converts internal simulation parameters into seeking/action vector for external portals.
        """
        action_vec = np.zeros(self.action_dim, dtype=np.float32)

        # Vector components derived from void gradient and node states
        action_vec[0] = float(np.tanh(void_gradient))
        action_vec[1] = float(np.tanh(abs(back_emf)))

        if nodes:
            avg_pos = np.mean([n.position for n in nodes.values()], axis=0)
            if len(avg_pos) >= 2:
                action_vec[2] = float(np.tanh(avg_pos[0]))
                action_vec[3] = float(np.tanh(avg_pos[1]))

        # Update digital twin registers
        self.external_registers["last_void_gradient"] = void_gradient
        self.external_registers["last_back_emf"] = back_emf
        self.external_registers["action_vector"] = action_vec
        self.external_registers["sync_timestamp"] = time.time()

        return action_vec


class InternalSimulationEngine:
    """
    [Internal Simulation Engine Core]
    Manages continuous breathing tick loop (while(1) / tick(dt)),
    virtual physics collision detection, tension wave dynamics, and digital twin sync.
    """
    def __init__(self, state_dim: int = 2, dt: float = 0.05):
        self.state_dim = state_dim
        self.dt = dt
        self.tick_count = 0

        self.collision_engine = VirtualPhysicsCollisionEngine(dim=state_dim)
        self.digital_twin_sync = DigitalTwinSyncInterface(action_dim=4)

        # Internal Nodes / Voxels Sandbox
        self.nodes: Dict[str, ConceptNode] = {
            "ALPHA": ConceptNode(node_id="ALPHA", position=np.array([-1.0, 0.0]), velocity=np.zeros(state_dim)),
            "BETA": ConceptNode(node_id="BETA", position=np.array([1.0, 0.0]), velocity=np.zeros(state_dim)),
        }

        # Background Thread Control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.void_gradient = 0.0
        self.back_emf = 0.0
        self.accumulated_tension = 0.0

    def tick(self, dt: Optional[float] = None, external_wave: Optional[np.ndarray] = None) -> SimulationStateReport:
        """
        Executes a single simulation tick:
        1. Updates positions via velocity integration
        2. Detects & resolves collisions (virtual physics)
        3. Dissipates tension waves & updates void gradients
        4. Syncs with Digital Twin registers
        """
        use_dt = dt or self.dt
        with self._lock:
            self.tick_count += 1

            # 1. Integrate position and velocity damping
            for node in self.nodes.values():
                node.position += node.velocity * use_dt
                node.velocity *= 0.95  # Physical damping
                node.phase = (node.phase + 2.0 * math.pi * use_dt) % (2.0 * math.pi)

            # 2. Virtual Physics Collisions
            events = self.collision_engine.detect_and_resolve_collisions(
                self.nodes, external_wave=external_wave, dt=use_dt
            )

            # 3. Tension Wave Dissipation
            current_tension = sum(n.tension for n in self.nodes.values())
            self.accumulated_tension = 0.9 * self.accumulated_tension + 0.1 * current_tension

            for node in self.nodes.values():
                node.tension *= 0.8  # Tension dissipation

            # Compute total kinetic + tension energy
            total_energy = sum(
                0.5 * node.mass * float(np.sum(node.velocity ** 2)) + node.tension
                for node in self.nodes.values()
            )

            # Void gradient directly linked to accumulated tension
            self.void_gradient = float(np.tanh(self.accumulated_tension * 0.5))

            # 4. Digital Twin Sync
            self.digital_twin_sync.sync_to_world(
                void_gradient=self.void_gradient,
                back_emf=self.back_emf,
                nodes=self.nodes,
            )

            report = SimulationStateReport(
                tick_count=self.tick_count,
                dt=use_dt,
                total_energy=total_energy,
                accumulated_tension=self.accumulated_tension,
                active_collisions_count=len(events),
                nodes_positions={k: v.position.copy() for k, v in self.nodes.items()},
                void_gradient_magnitude=self.void_gradient,
            )

            return report

    def start_background_loop(self):
        """Starts the continuous background tick loop thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def stop_background_loop(self):
        """Stops the continuous background tick loop thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _background_loop(self):
        while self._running:
            self.tick(dt=self.dt)
            time.sleep(self.dt)
