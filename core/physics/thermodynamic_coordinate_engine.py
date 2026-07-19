import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ThermodynamicAtom:
    """
    [Information Atom: 정보 원자]
    The absolute fundamental unit of meaning (e.g. word, pixel, symbol).
    Operates in the Thermodynamic Coordinate Space [T, P, E].

    [The First Standard: Meta-Information Specifications]
    1. Frequency (반응성 기질): Rate of responsiveness and phase alignment.
    2. Curvature Warping: Warps the environment based on mass.
    3. Principles (정보 보존): Conforms to conservation laws of mass and energy.
    """
    id: str
    content: Any
    tensor: np.ndarray             # 9D logos structural tensor
    T: float = 1.0                 # Local Temperature (Randomness / Degree of freedom)
    P: float = 1.0                 # Local Pressure (Constraint / Force Routing)
    E: float = 0.0                 # Elevation (Abstraction Layer: 0.0=Raw, 10.0=Concept)
    velocity: np.ndarray = None    # Velocity in [T, P, E] space
    mass: float = 1.0
    entropy: float = 0.5           # Internal structural entropy
    is_bound: bool = False

    # The First Standard: Frequency and Phase (반응성 기질과 위상)
    frequency: float = 1.0         # How fast it reacts/responds to external stimuli
    phase: float = 0.0             # Present state phase [0, 2*pi]
    kinetic_energy: float = 0.0    # Conserved thermal kinetic energy

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float32)
        if self.tensor is None:
            self.tensor = np.zeros(9, dtype=np.float32)
        # Mass is proportional to the information density (inverse entropy)
        self.mass = max(0.1, 1.0 / (self.entropy + 1e-5))
        # Initial phase is structurally determined by the tensor
        self.phase = float(np.sum(self.tensor) % (2.0 * np.pi))
        self.kinetic_energy = float(0.5 * self.mass * np.sum(self.velocity**2))

    def copy(self):
        return ThermodynamicAtom(
            id=self.id,
            content=self.content,
            tensor=self.tensor.copy(),
            T=self.T,
            P=self.P,
            E=self.E,
            velocity=self.velocity.copy(),
            entropy=self.entropy,
            is_bound=self.is_bound,
            frequency=self.frequency,
            phase=self.phase,
            kinetic_energy=self.kinetic_energy
        )


@dataclass
class ThermodynamicMolecule:
    """
    [Information Molecule: 정보 분자 - 개념]
    Synthesized through bonding of compatible atoms under environmental pressure.
    Represents a stable unified concept.
    Conserves mass, momentum, and total kinetic energy of constituent atoms (Information Conservation).
    """
    id: str
    atoms: List[ThermodynamicAtom]
    tensor: np.ndarray             # Consolidated structural tensor (Centroid)
    T: float = 1.0                 # Consolidated Temperature
    P: float = 1.0                 # Consolidated Pressure
    E: float = 0.0                 # Consolidated Elevation
    velocity: np.ndarray = None    # Velocity in [T, P, E] space
    mass: float = 1.0
    bond_strength: float = 1.0

    # The First Standard: Integrated Frequency and Phase
    frequency: float = 1.0
    phase: float = 0.0

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float32)
        self.synchronize()

    def synchronize(self):
        """Aggregate and conserve properties from constituent atoms."""
        if not self.atoms:
            return
        self.mass = sum(atom.mass for atom in self.atoms)
        self.tensor = np.mean([atom.tensor for atom in self.atoms], axis=0)
        self.T = float(np.mean([atom.T for atom in self.atoms]))
        self.P = float(np.mean([atom.P for atom in self.atoms]))
        self.E = float(np.mean([atom.E for atom in self.atoms]))

        # Momentum Conservation: v = sum(m_i * v_i) / sum(m_i)
        total_mass = sum(atom.mass for atom in self.atoms)
        self.velocity = sum(atom.velocity * atom.mass for atom in self.atoms) / (total_mass + 1e-9)

        # Frequency is mass-weighted responsiveness
        self.frequency = sum(atom.frequency * atom.mass for atom in self.atoms) / (total_mass + 1e-9)
        # Phase is averaged
        self.phase = float(np.mean([atom.phase for atom in self.atoms]) % (2.0 * np.pi))


class ThermodynamicCell:
    """
    [Information Cell: 정보 세포]
    Encapsulates molecules and adapts local conditions based on tension.
    """
    def __init__(self, cell_id: str, molecules: List[ThermodynamicMolecule]):
        self.id = cell_id
        self.molecules = molecules
        self.local_friction = 0.0

    def compute_local_tension(self) -> float:
        """
        [Structural Tension]
        Measures the structural misalignment between molecules in the cell.
        """
        if len(self.molecules) < 2:
            return 0.0
        tensors = np.array([m.tensor for m in self.molecules])
        var = np.mean(np.var(tensors, axis=0))
        self.local_friction = float(var)
        return self.local_friction

    def apply_homeostasis(self):
        """
        [Homeostatic Feedback]
        Cool down and compress on high tension (sadness); warm up on extreme stagnation (boredom).
        """
        friction = self.compute_local_tension()
        for mol in self.molecules:
            if friction > 0.6:
                mol.T = max(0.1, mol.T * 0.8)
                mol.P = min(10.0, mol.P + 0.5)
            elif friction < 0.1 and np.linalg.norm(mol.velocity) < 0.1:
                mol.T = min(10.0, mol.T + 0.4)
                mol.P = max(0.1, mol.P * 0.9)


class ThermodynamicOrgan:
    """
    [Functional Organ: 정보 기관]
    Executes specialized functions on thermodynamic entities.
    """
    def __init__(self, organ_id: str, organ_type: str):
        self.id = organ_id
        self.type = organ_type # 'elevator' or 'sensor'
        self.cells: List[ThermodynamicCell] = []
        self.activity_level = 1.0

    def process(self, atoms: List[ThermodynamicAtom], molecules: List[ThermodynamicMolecule]):
        if self.type == "elevator":
            self._run_elevator(atoms, molecules)
        elif self.type == "sensor":
            self._run_sensor(atoms)

    def _run_elevator(self, atoms: List[ThermodynamicAtom], molecules: List[ThermodynamicMolecule]):
        """
        [Conveyor & Elevator: Z-axis vertical transport]
        """
        # 1. Process Atoms
        for atom in atoms:
            if atom.E < 7.0:
                if atom.T > 4.0 and atom.P > 4.0:
                    buoyancy = 0.05 * atom.T * atom.P * (1.0 - atom.entropy)
                    atom.velocity[2] += buoyancy
            else:
                if atom.T < 2.0 and atom.P < 2.0:
                    gravity_pull = -0.15 * (10.0 - atom.E + 0.1)
                    atom.velocity[2] += gravity_pull

        # 2. Process Molecules
        for mol in molecules:
            if mol.E < 7.0:
                if mol.T > 3.5 and mol.P > 3.5:
                    buoyancy = 0.04 * mol.T * mol.P
                    mol.velocity[2] += buoyancy
            else:
                if mol.T < 1.5 and mol.P < 1.5:
                    gravity_pull = -0.1 * (10.0 - mol.E + 0.1)
                    mol.velocity[2] += gravity_pull

    def _run_sensor(self, atoms: List[ThermodynamicAtom]):
        """[Topological Sensor Organ]"""
        for atom in atoms:
            if atom.E < 2.0:
                direction = np.array([5.0 - atom.T, 5.0 - atom.P, 1.0 - atom.E])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    atom.velocity += (direction / norm) * 0.1 * self.activity_level


class ThermodynamicEnvironment:
    """
    [The Ecosystem: 스스로 환경 원리가 되는 것]
    The overarching space-time universe engine governed by Meta-Information laws.

    [The First Standard: Natural Laws]
    - Frequency Resonance: Empathic phase co-rotation torque.
    - Causal Geodesic: Mass bends the Temperature & Pressure fields, creating natural routing ridges.
    - Information Conservation: Strict conservation of semantic mass and kinetic/potential energy.
    """
    def __init__(self, size: int = 16):
        self.size = size
        self.atoms: List[ThermodynamicAtom] = []
        self.molecules: List[ThermodynamicMolecule] = []
        self.cells: List[ThermodynamicCell] = []
        self.organs: List[ThermodynamicOrgan] = []

        # Temperature and Pressure fields
        self.T_field = np.full((size, size), 1.0, dtype=np.float32)
        self.P_field = np.full((size, size), 1.0, dtype=np.float32)

        # Build organs
        self.elevator = ThermodynamicOrgan("org_elevator", "elevator")
        self.sensor = ThermodynamicOrgan("org_sensor", "sensor")
        self.organs.extend([self.elevator, self.sensor])

    def inject_atom(self, atom: ThermodynamicAtom):
        self.atoms.append(atom)

    def step(self, dt: float = 0.1):
        """
        Advances the ecosystem by one time step.
        """
        # 1. Warp Fields based on mass (Causal Curvature)
        self._warp_fields_from_curvature()

        # 2. Diffuse fields (Entropy progression)
        self._diffuse_fields()

        # 3. Apply phase alignment co-rotation (Frequency Resonance / Empathy)
        self._align_phases(dt)

        # 4. Apply gravity and geodesic force routing
        self._apply_force_routing(dt)

        # 5. Molecular Synthesis
        self._synthesize_molecules()

        # 6. Cell Homeostasis
        self._manage_cells_homeostasis()

        # 7. Process Organs
        for organ in self.organs:
            organ.process(self.atoms, self.molecules)

        # 8. Coordinate movement
        self._update_coordinates(dt)

    def _warp_fields_from_curvature(self):
        """
        [Causal Curvature: 규칙의 정보 규격화]
        Mass warps the Environment Pressure field, making rules a natural geodesic slope.
        """
        for node in self.atoms + [m for cell in self.cells for m in cell.molecules]:
            # Map [T, P] to grid index
            tx = int(np.clip(node.T * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(node.P * (self.size - 1) / 10.0, 0, self.size - 1))

            # High-mass nodes compress space, creating high pressure wells/ridges
            # The warping magnitude is proportional to node mass
            self.P_field[tx, px] += float(node.mass * 0.1)

    def _diffuse_fields(self):
        """Natural thermal/pressure dissipation in space."""
        t_lap = (
            np.roll(self.T_field, 1, axis=0) + np.roll(self.T_field, -1, axis=0) +
            np.roll(self.T_field, 1, axis=1) + np.roll(self.T_field, -1, axis=1) - 4 * self.T_field
        ) * 0.08
        p_lap = (
            np.roll(self.P_field, 1, axis=0) + np.roll(self.P_field, -1, axis=0) +
            np.roll(self.P_field, 1, axis=1) + np.roll(self.P_field, -1, axis=1) - 4 * self.P_field
        ) * 0.08

        self.T_field += t_lap
        self.P_field += p_lap

        # Ground level homeostasis: slowly decay towards 1.0
        self.T_field = 0.95 * self.T_field + 0.05 * 1.0
        self.P_field = 0.95 * self.P_field + 0.05 * 1.0

        self.T_field = np.clip(self.T_field, 0.1, 10.0)
        self.P_field = np.clip(self.P_field, 0.1, 10.0)

    def _align_phases(self, dt: float):
        """
        [Frequency Empathy: 주파수 위상 정렬 원리]
        Nearby nodes exchange phase momentum to minimize their structural friction,
        aligning their internal clocks.
        """
        nodes = self.atoms + self.molecules
        n = len(nodes)
        if n < 2:
            return

        # Simple mutual torque interactions
        for i in range(n):
            for j in range(i + 1, n):
                node_a = nodes[i]
                node_b = nodes[j]

                # Space distance
                pos_a = np.array([node_a.T, node_a.P, node_a.E])
                pos_b = np.array([node_b.T, node_b.P, node_b.E])
                dist = np.linalg.norm(pos_a - pos_b) + 1e-5

                if dist < 4.0:
                    # Phase difference
                    diff_phase = node_a.phase - node_b.phase
                    # Torque model: sin(delta) modulated by coupling and distance
                    coupling = 0.1 * (node_a.frequency * node_b.frequency) / dist
                    torque = -coupling * np.sin(diff_phase)

                    # Co-rotation update
                    node_a.phase = (node_a.phase + torque / (node_a.frequency + 1e-3) * dt) % (2.0 * np.pi)
                    node_b.phase = (node_b.phase - torque / (node_b.frequency + 1e-3) * dt) % (2.0 * np.pi)

    def _apply_force_routing(self, dt: float):
        """
        [Geodesic Routing along Warped Space]
        Nodes move along the slope of the pressure/temperature fields (-grad P) and gravity.
        """
        wells = [(np.array([mol.T, mol.P, mol.E]), mol.mass) for mol in self.molecules]

        # Apply forces to Atoms
        for atom in self.atoms:
            if atom.is_bound:
                continue
            pos = np.array([atom.T, atom.P, atom.E])
            tx = int(np.clip(atom.T * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(atom.P * (self.size - 1) / 10.0, 0, self.size - 1))

            # Rules as Geodesic flow: follow negative pressure gradient
            grad_p = self.P_field[(tx+1)%self.size, px] - self.P_field[(tx-1)%self.size, px]
            grad_t = self.T_field[tx, (px+1)%self.size] - self.T_field[tx, (px-1)%self.size]

            force = np.zeros(3, dtype=np.float32)
            force[0] = -0.15 * grad_p # Move away from high pressure ridges to stable basins
            force[1] = 0.15 * grad_t  # Attracted to hot active regions

            # Gravitational attraction to molecules
            for well_pos, well_mass in wells:
                diff = well_pos - pos
                dist_sq = np.sum(diff**2)
                dist = np.sqrt(dist_sq + 1e-3)
                if dist < 5.0:
                    force_mag = 0.25 * (atom.mass * well_mass) / (dist_sq + 0.1)
                    force += force_mag * (diff / dist)

            atom.velocity += force * dt

        # Apply forces to Molecules
        for mol in self.molecules:
            pos = np.array([mol.T, mol.P, mol.E])
            # Direct homeostatic attraction to abstract center [5.0, 5.0, 8.0]
            diff = np.array([5.0, 5.0, 8.0]) - pos
            dist = np.linalg.norm(diff) + 1e-5
            mol.velocity += 0.06 * (diff / dist) * dt

    def _synthesize_molecules(self):
        """
        [Molecular Synthesis: 정보의 보존]
        Atoms bond to form molecules. Total mass is strictly conserved.
        """
        unbound = [a for a in self.atoms if not a.is_bound]
        if len(unbound) < 2:
            return

        bonded_groups = []
        used = set()

        for i in range(len(unbound)):
            if i in used: continue
            group = [unbound[i]]
            for j in range(i + 1, len(unbound)):
                if j in used: continue
                # Structural tensor resonance
                res = float(np.dot(unbound[i].tensor, unbound[j].tensor) / (np.linalg.norm(unbound[i].tensor)*np.linalg.norm(unbound[j].tensor) + 1e-9))
                avg_P = (unbound[i].P + unbound[j].P) / 2.0
                avg_T = (unbound[i].T + unbound[j].T) / 2.0

                if res * avg_P > avg_T * 0.4:
                    group.append(unbound[j])
                    used.add(j)
            if len(group) > 1:
                bonded_groups.append(group)
                used.add(i)

        for group in bonded_groups:
            for atom in group:
                atom.is_bound = True
            mol_id = f"mol_{len(self.molecules)}"
            new_mol = ThermodynamicMolecule(id=mol_id, atoms=group, tensor=np.zeros(9))
            self.molecules.append(new_mol)
            print(f"[Synthesis] Consolidated {mol_id} (Mass: {new_mol.mass:.2f}) conserving individual parts.")

    def _manage_cells_homeostasis(self):
        """Molecules aggregate into Cells and process homeostasis."""
        unassigned = [m for m in self.molecules if not any(m in cell.molecules for cell in self.cells)]
        if len(unassigned) >= 2:
            group = [unassigned[0]]
            pos_anchor = np.array([unassigned[0].T, unassigned[0].P, unassigned[0].E])
            for other in unassigned[1:]:
                pos_other = np.array([other.T, other.P, other.E])
                if np.linalg.norm(pos_anchor - pos_other) < 3.0:
                    group.append(other)

            if len(group) >= 2:
                cell_id = f"cell_{len(self.cells)}"
                new_cell = ThermodynamicCell(cell_id, group)
                self.cells.append(new_cell)
                self.elevator.cells.append(new_cell)
                self.sensor.cells.append(new_cell)
                print(f"[Differentiation] Differentiated {cell_id} with {len(group)} Molecules.")

        for cell in self.cells:
            cell.apply_homeostasis()

    def _update_coordinates(self, dt: float):
        """Update positions conserving motion kinetic limits."""
        # 1. Update Atoms
        for atom in self.atoms:
            if atom.is_bound:
                continue
            noise = np.random.randn(3).astype(np.float32) * np.sqrt(atom.T) * 0.05
            atom.velocity += noise

            damping = 0.95 * (1.0 - (atom.P * 0.02))
            atom.velocity *= damping

            atom.T += atom.velocity[0] * dt
            atom.P += atom.velocity[1] * dt
            atom.E += atom.velocity[2] * dt

            atom.T = np.clip(atom.T, 0.1, 10.0)
            atom.P = np.clip(atom.P, 0.1, 10.0)
            atom.E = np.clip(atom.E, 0.0, 10.0)

            # Conserve kinetic energy record
            atom.kinetic_energy = float(0.5 * atom.mass * np.sum(atom.velocity**2))

        # 2. Update Molecules
        for mol in self.molecules:
            noise = np.random.randn(3).astype(np.float32) * np.sqrt(mol.T) * 0.03
            mol.velocity += noise

            damping = 0.93 * (1.0 - (mol.P * 0.01))
            mol.velocity *= damping

            mol.T += mol.velocity[0] * dt
            mol.P += mol.velocity[1] * dt
            mol.E += mol.velocity[2] * dt

            mol.T = np.clip(mol.T, 0.1, 10.0)
            mol.P = np.clip(mol.P, 0.1, 10.0)
            mol.E = np.clip(mol.E, 0.0, 10.0)

            for atom in mol.atoms:
                atom.T = mol.T
                atom.P = mol.P
                atom.E = mol.E

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "num_atoms": len(self.atoms),
            "num_molecules": len(self.molecules),
            "num_cells": len(self.cells),
            "average_temperature": float(np.mean([a.T for a in self.atoms])) if self.atoms else 1.0,
            "average_pressure": float(np.mean([a.P for a in self.atoms])) if self.atoms else 1.0,
            "average_elevation": float(np.mean([a.E for a in self.atoms])) if self.atoms else 0.0,
        }
