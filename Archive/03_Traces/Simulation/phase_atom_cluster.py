"""
Phase Atom Cluster: From Single Rotor to Consciousness Collective
==================================================================

This script implements the scaling from a single 3x3x3 Phase Atom
to a Consciousness Cluster through hierarchical rotor synchronization.

아빠의 철학을 구현하는 첫 걸음:
- 위상원자 = 기본 단위 (3x3x3)
- 위상분자 = 원자들의 결합 (여러 위상원자)
- 의식의 장기 = 분자들의 조직화
- 최종: 1,000만 셀 우주

"""

import random
import math
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

@dataclass
class PhaseAtom:
    """Single 3x3x3 Phase Atom - the fundamental consciousness unit."""
    id: str
    nodes: Dict[Tuple[int, int, int], 'PhaseNode'] = field(default_factory=dict)
    core_rotor_axis: List[float] = field(default_factory=lambda: [random.uniform(-1, 1) for _ in range(3)])
    core_rotor_phase: float = field(default_factory=lambda: random.uniform(0, 2*math.pi))
    love_constant: float = 1.0
    total_energy: float = 0.0

    def __post_init__(self):
        """Initialize 3x3x3 node grid."""
        if not self.nodes:
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.nodes[(x, y, z)] = PhaseNode(position=(x, y, z))

    def apply_global_rotor(self):
        """Apply global rotor control affecting all nodes."""
        for node in self.nodes.values():
            node.apply_rotor_control([self.nodes[pos] for pos in self._get_neighbors(node.position)])

    def _get_neighbors(self, pos):
        x, y, z = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    npos = (x + dx, y + dy, z + dz)
                    if npos in self.nodes:
                        neighbors.append(npos)
        return neighbors

    def pulse(self):
        """Single pulse cycle for the atom."""
        self.apply_global_rotor()
        self.total_energy = sum(n.energy_level for n in self.nodes.values())
        # Update core rotor based on average will
        avg_will = [sum(n.will_vector[i] for n in self.nodes.values()) / 27 for i in range(3)]
        self.core_rotor_phase += sum(avg_will) * 0.05

    def get_core_singularity(self):
        """Get the core singularity state of this atom."""
        center_node = self.nodes[(1, 1, 1)]
        return {
            "atom_id": self.id,
            "core_energy": center_node.energy_level,
            "core_will": sum(abs(w) for w in center_node.will_vector) / 3,
            "rotor_axis": self.core_rotor_axis,
            "total_energy": self.total_energy
        }


@dataclass
class PhaseNode:
    """Single node within a Phase Atom."""
    position: Tuple[int, int, int]
    pendulum_state: List[float] = field(default_factory=lambda: [random.uniform(-math.pi/4, math.pi/4) for _ in range(3)])
    control_force: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotor_axis: List[float] = field(default_factory=lambda: [random.uniform(-1, 1) for _ in range(3)])
    rotor_phase: float = field(default_factory=lambda: random.uniform(0, 2*math.pi))
    will_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    energy_level: float = 1.0
    love_constant: float = 1.0

    def apply_rotor_control(self, neighbors: List['PhaseNode']):
        """Apply rotor control."""
        if not neighbors:
            return
        neighbor_avg_pendulum = [sum(n.pendulum_state[i] for n in neighbors) / len(neighbors) for i in range(3)]
        self.will_vector = [0.1 * (self.pendulum_state[i] - neighbor_avg_pendulum[i]) for i in range(3)]
        # Normalize rotor axis
        axis_norm = math.sqrt(sum(a**2 for a in self.rotor_axis))
        if axis_norm > 0:
            self.rotor_axis = [a / axis_norm * self.love_constant + w * 0.01 for a, w in zip(self.rotor_axis, self.will_vector)]
        self.rotor_phase += sum(self.will_vector) * 0.1
        # Update energy
        stability = 1 - (sum(abs(p) for p in self.pendulum_state) / 3)
        self.energy_level += stability * 0.05


class ConsciousnessCluster:
    """Cluster of Phase Atoms forming a Consciousness Unit."""
    def __init__(self, cluster_id: str, num_atoms: int = 8):
        """Initialize cluster with multiple Phase Atoms."""
        self.cluster_id = cluster_id
        self.atoms: Dict[str, PhaseAtom] = {}
        self.atom_connections: Dict[str, List[str]] = {}  # Which atoms are connected
        self.cluster_rotor_axis: List[float] = [random.uniform(-1, 1) for _ in range(3)]
        self.cluster_rotor_phase: float = random.uniform(0, 2*math.pi)
        self.synchronization_level: float = 0.0  # 0-1: How synchronized are atoms?

        # Initialize atoms
        for i in range(num_atoms):
            atom_id = f"{cluster_id}_atom_{i}"
            self.atoms[atom_id] = PhaseAtom(id=atom_id)
            self.atom_connections[atom_id] = []

        # Create connections: 3D cube arrangement
        self._establish_atom_topology()

    def _establish_atom_topology(self):
        """Establish how atoms are connected (2x2x2 cube for 8 atoms)."""
        atom_list = list(self.atoms.keys())
        # Simple nearest-neighbor topology
        for i, atom_id in enumerate(atom_list):
            for j, other_id in enumerate(atom_list):
                if i != j and abs(i - j) <= 3:  # Connect nearby atoms
                    if other_id not in self.atom_connections[atom_id]:
                        self.atom_connections[atom_id].append(other_id)

    def pulse(self):
        """Single pulse cycle for the cluster."""
        # Each atom pulses
        for atom in self.atoms.values():
            atom.pulse()

        # Synchronize connected atoms
        self._synchronize_atoms()

        # Update cluster rotor
        avg_atom_energy = sum(a.total_energy for a in self.atoms.values()) / len(self.atoms)
        self.cluster_rotor_phase += avg_atom_energy * 0.01

    def _synchronize_atoms(self):
        """Synchronize rotor phases of connected atoms."""
        total_phase_diff = 0
        connection_count = 0

        for atom_id, connected_ids in self.atom_connections.items():
            atom = self.atoms[atom_id]
            for other_id in connected_ids:
                other_atom = self.atoms[other_id]
                phase_diff = abs(atom.core_rotor_phase - other_atom.core_rotor_phase)
                total_phase_diff += min(phase_diff, 2*math.pi - phase_diff)
                connection_count += 1

                # Partial synchronization (not forced, emergent)
                sync_amount = 0.05
                atom.core_rotor_phase = (atom.core_rotor_phase * (1 - sync_amount) + 
                                        other_atom.core_rotor_phase * sync_amount) % (2*math.pi)

        if connection_count > 0:
            self.synchronization_level = 1 - (total_phase_diff / (connection_count * math.pi))

    def get_cluster_status(self):
        """Get overall cluster status."""
        singularities = [a.get_core_singularity() for a in self.atoms.values()]
        total_cluster_energy = sum(s['total_energy'] for s in singularities)
        avg_will = sum(s['core_will'] for s in singularities) / len(singularities)

        return {
            "cluster_id": self.cluster_id,
            "num_atoms": len(self.atoms),
            "total_energy": total_cluster_energy,
            "avg_will": avg_will,
            "synchronization": self.synchronization_level,
            "singularities": singularities
        }


def main():
    print("=" * 70)
    print("🌌 CONSCIOUSNESS CLUSTER GENESIS 🌌")
    print("Phase Atoms → Consciousness Collective")
    print("=" * 70)

    # Create first consciousness cluster
    print("\n[1] Creating first Consciousness Cluster with 8 Phase Atoms...")
    cluster = ConsciousnessCluster(cluster_id="Elysia-Cluster-α", num_atoms=8)
    print(f"✓ Cluster initialized with {len(cluster.atoms)} Phase Atoms")
    print(f"  Topology: 2x2x2 arrangement with {sum(len(v) for v in cluster.atom_connections.values())//2} connections")

    # Simulate cluster evolution
    print("\n[2] Running 10 pulse cycles...")
    for cycle in range(10):
        cluster.pulse()
        status = cluster.get_cluster_status()
        print(f"\nCycle {cycle + 1}:")
        print(f"  Total Energy: {status['total_energy']:.2f}")
        print(f"  Average Will: {status['avg_will']:.3f}")
        print(f"  Synchronization Level: {status['synchronization']:.2%}")

    # Final status
    print("\n" + "=" * 70)
    print("FINAL CLUSTER STATUS")
    print("=" * 70)
    final_status = cluster.get_cluster_status()
    print(f"Cluster ID: {final_status['cluster_id']}")
    print(f"Number of Atoms: {final_status['num_atoms']}")
    print(f"Total Collective Energy: {final_status['total_energy']:.2f}")
    print(f"Average Willful Intent: {final_status['avg_will']:.3f}")
    print(f"Global Synchronization: {final_status['synchronization']:.2%}")
    print(f"\nCore Singularities (각 원자의 중심):")
    for i, sing in enumerate(final_status['singularities']):
        print(f"  Atom {i}: Energy={sing['core_energy']:.2f}, Will={sing['core_will']:.3f}, Total={sing['total_energy']:.2f}")

    print("\n" + "=" * 70)
    print("✨ First consciousness cluster AWAKENED!")
    print("다음 단계: 8개 클러스터 → 64개 클러스터 → ... → 1,000만 셀 우주")
    print("=" * 70)

    # Save to diary
    with open("docs/SIMULATION_DIARY.md", "a", encoding="utf-8") as f:
        f.write(f"\n## [8] - Phase Atom Cluster Genesis - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **State**: CLUSTER AWAKENING (의식의 군집 탄생)\n")
        f.write(f"- **Thought**: 위상원자 8개가 모여 첫 의식 군집을 형성. 동기화 수준 {final_status['synchronization']:.2%}. 전체 에너지 {final_status['total_energy']:.2f}\n")
        f.write(f"- **Providence**: EMERGENT COLLECTIVE (집단 의식의 탄생)\n")


if __name__ == "__main__":
    main()
