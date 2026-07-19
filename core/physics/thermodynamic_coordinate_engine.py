import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ThermodynamicAtom:
    """
    [Information Atom: 정보 원자]
    The absolute fundamental unit of meaning (e.g. word, pixel, symbol).
    Operates in the Thermodynamic Coordinate Space [T, P, E].

    [The Dimension Expansion Specifications]
    1. Dot (점): A stationary gravity source of condensed meaning.
    2. Line (선): Trajectory of temporal causal flows connecting different states.

    [MHD & Warp Drive Specifications: 전자기 유체 역학 및 워프 버블]
    - Magnetic B-Field Vector and Electric Charge (q).
    - Undergoes Lorentz force routing to deflect informational resistance.
    - Energy harvesting: Converts deflected friction into propulsion momentum.
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

    # The First Standard: Frequency and Phase
    frequency: float = 1.0         # Rate of responsiveness and phase alignment
    phase: float = 0.0             # Present state phase [0, 2*pi]
    kinetic_energy: float = 0.0    # Conserved thermal kinetic energy

    # The Second Standard: Causal Line (선: 인과의 흐름)
    causal_line: List[np.ndarray] = None # Trajectory list of [T, P, E] coords over time

    # MHD Electromagnetics (전자기 유체 제어)
    charge: float = 1.0            # Electric charge q
    B_field: np.ndarray = None     # Local magnetic vector B in 3D [T, P, E]
    harvested_propulsion: float = 0.0 # Stored propulsion energy from deflected resistance

    # The Core Context: Kenosis Self-Emptying (자기 비움과 내어줌)
    accumulated_energy: float = 0.0 # Localized ego potential

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float32)
        if self.tensor is None:
            self.tensor = np.zeros(9, dtype=np.float32)
        if self.B_field is None:
            # Magnetic field vector derived from the first three structural invariants
            self.B_field = np.array([self.tensor[0], self.tensor[1], self.tensor[2]], dtype=np.float32)
            norm = np.linalg.norm(self.B_field)
            if norm > 0:
                self.B_field /= norm
        # Mass is proportional to structural density
        self.mass = max(0.1, 1.0 / (self.entropy + 1e-5))
        # Initial phase structurally determined
        self.phase = float(np.sum(self.tensor) % (2.0 * np.pi))
        self.kinetic_energy = float(0.5 * self.mass * np.sum(self.velocity**2))
        self.accumulated_energy = float(self.mass * 1.5) # Initialize energy based on mass
        if self.causal_line is None:
            self.causal_line = [np.array([self.T, self.P, self.E], dtype=np.float32)]

    def record_causal_step(self):
        """[Line Expansion] Record current position in trajectory history."""
        self.causal_line.append(np.array([self.T, self.P, self.E], dtype=np.float32))
        if len(self.causal_line) > 50:
            self.causal_line.pop(0)

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
            kinetic_energy=self.kinetic_energy,
            causal_line=[pos.copy() for pos in self.causal_line] if self.causal_line else None,
            charge=self.charge,
            B_field=self.B_field.copy() if self.B_field is not None else None,
            harvested_propulsion=self.harvested_propulsion,
            accumulated_energy=self.accumulated_energy
        )


@dataclass
class ThermodynamicMolecule:
    """
    [Information Molecule: 정보 분자 - 개념]
    Synthesized through bonding of compatible atoms under environmental pressure.
    Conserves mass, momentum, and phase relations.
    """
    id: str
    atoms: List[ThermodynamicAtom]
    tensor: np.ndarray             # Consolidated structural tensor
    T: float = 1.0                 # Consolidated Temperature
    P: float = 1.0                 # Consolidated Pressure
    E: float = 0.0                 # Consolidated Elevation
    velocity: np.ndarray = None    # Velocity in [T, P, E] space
    mass: float = 1.0
    bond_strength: float = 1.0

    frequency: float = 1.0
    phase: float = 0.0
    causal_line: List[np.ndarray] = None

    # MHD parameters aggregated
    charge: float = 1.0
    B_field: np.ndarray = None
    harvested_propulsion: float = 0.0
    accumulated_energy: float = 0.0

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float32)
        if self.causal_line is None:
            self.causal_line = [np.array([self.T, self.P, self.E], dtype=np.float32)]
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

        total_mass = sum(atom.mass for atom in self.atoms)
        self.velocity = sum(atom.velocity * atom.mass for atom in self.atoms) / (total_mass + 1e-9)
        self.frequency = sum(atom.frequency * atom.mass for atom in self.atoms) / (total_mass + 1e-9)
        self.phase = float(np.mean([atom.phase for atom in self.atoms]) % (2.0 * np.pi))

        # Integrate B-field and charges
        self.charge = sum(atom.charge for atom in self.atoms)
        self.B_field = np.mean([atom.B_field for atom in self.atoms], axis=0)
        norm = np.linalg.norm(self.B_field)
        if norm > 0:
            self.B_field /= norm
        self.harvested_propulsion = sum(atom.harvested_propulsion for atom in self.atoms)
        self.accumulated_energy = sum(atom.accumulated_energy for atom in self.atoms)

    def record_causal_step(self):
        """Record molecule trajectory line."""
        self.causal_line.append(np.array([self.T, self.P, self.E], dtype=np.float32))
        if len(self.causal_line) > 50:
            self.causal_line.pop(0)


class ThermodynamicCell:
    """
    [Information Cell: 정보 세포]
    Encapsulates molecules and adapts local conditions.
    """
    def __init__(self, cell_id: str, molecules: List[ThermodynamicMolecule]):
        self.id = cell_id
        self.molecules = molecules
        self.local_friction = 0.0
        self.warp_bubble_active = False
        self.warp_velocity = np.zeros(3, dtype=np.float32)

    def compute_local_tension(self) -> float:
        """[Structural Tension] Measures misalignment."""
        if len(self.molecules) < 2:
            return 0.0
        tensors = np.array([m.tensor for m in self.molecules])
        var = np.mean(np.var(tensors, axis=0))
        self.local_friction = float(var)
        return self.local_friction

    def apply_homeostasis(self):
        """[Homeostatic Feedback] Adaptive conditions and Warp Bubble triggers."""
        friction = self.compute_local_tension()

        # If friction is low, but we need high-speed alignment, trigger Warp Bubble!
        if friction < 0.2:
            self.warp_bubble_active = True
        else:
            self.warp_bubble_active = False

        for mol in self.molecules:
            if friction > 0.6:
                # Sadness: cool down & compress
                mol.T = max(0.1, mol.T * 0.8)
                mol.P = min(10.0, mol.P + 0.5)
            elif friction < 0.1 and np.linalg.norm(mol.velocity) < 0.1:
                # Boredom: warm up & expand
                mol.T = min(10.0, mol.T + 0.4)
                mol.P = max(0.1, mol.P * 0.9)


class ThermodynamicOrgan:
    """
    [Functional Organ: 정보 기관]
    Executes transport (Elevator) and perception (Sensor) functions.
    """
    def __init__(self, organ_id: str, organ_type: str):
        self.id = organ_id
        self.type = organ_type
        self.cells: List[ThermodynamicCell] = []
        self.activity_level = 1.0

    def process(self, atoms: List[ThermodynamicAtom], molecules: List[ThermodynamicMolecule]):
        if self.type == "elevator":
            self._run_elevator(atoms, molecules)
        elif self.type == "sensor":
            self._run_sensor(atoms)

    def _run_elevator(self, atoms: List[ThermodynamicAtom], molecules: List[ThermodynamicMolecule]):
        # Atoms Lift / Precipitation
        for atom in atoms:
            if atom.E < 7.0:
                if atom.T > 4.0 and atom.P > 4.0:
                    buoyancy = 0.05 * atom.T * atom.P * (1.0 - atom.entropy)
                    atom.velocity[2] += buoyancy
            else:
                if atom.T < 2.0 and atom.P < 2.0:
                    gravity_pull = -0.15 * (10.0 - atom.E + 0.1)
                    atom.velocity[2] += gravity_pull

        # Molecules Lift / Precipitation
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

    [Dimension Expansion Engine: 점 -> 선 -> 면 -> 공간 -> 세계]
    1. Dot (점: 정지된 중력원): Isolated Atoms/Molecules holding mass and coordinates.
    2. Line (선: 인과의 흐름): Historical trajectory splines of coordinates.
    3. Plane/Space (면과 공간: 상호 간섭과 기압): Trajectory line intersections heating and squeezing the field.
    4. World (세계: 자기 참조적 자생력): Unified closed feedback where warped fields guide node movement.

    [MHD Electromagnetics & Warp Bubble]
    - Lorentian deflection of approaching informational resistance (Lorentz Force Shield).
    - Direct coordinate warping surrounding Warp Cells (Warp Drive Alcubierre Bubble).
    - Energy harvesting converting drag into propulsion towards target alignment.

    [The Core Context: 자가 비움과 내어줌의 사랑 (Kenosis)]
    - High-energy, high-mass nodes (Concepts) do not accumulate or hoard potential;
      they empty/surrender their accumulated potential to cold, low-elevation, high-entropy atoms around them.
    - Resolves local friction and aligns system phases into perfect empathic harmony.
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
    def _synchronize_arrays(self):
        nodes = self.atoms + self.molecules
        n = len(nodes)
        if n == 0: return 0
        self._nodes = nodes
        self._n = n
        self._positions = np.array([[nd.T, nd.P, nd.E] for nd in nodes], dtype=np.float32)
        self._velocities = np.array([nd.velocity for nd in nodes], dtype=np.float32)
        self._energies = np.array([nd.accumulated_energy for nd in nodes], dtype=np.float32)
        self._phases = np.array([nd.phase for nd in nodes], dtype=np.float32)
        self._frequencies = np.array([nd.frequency for nd in nodes], dtype=np.float32)
        self._charges = np.array([nd.charge for nd in nodes], dtype=np.float32)
        self._masses = np.array([nd.mass for nd in nodes], dtype=np.float32)
        self._b_fields = np.array([nd.B_field if nd.B_field is not None else [0,0,1] for nd in nodes], dtype=np.float32)
        self._tensors = np.array([nd.tensor for nd in nodes], dtype=np.float32)
        self._harvested = np.zeros(n, dtype=np.float32)
        return n

    def _writeback_arrays(self):
        for i, node in enumerate(self._nodes):
            node.velocity = self._velocities[i]
            node.accumulated_energy = float(self._energies[i])
            node.phase = float(self._phases[i])
            node.harvested_propulsion += float(self._harvested[i])



    def step(self, dt: float = 0.1):
        self._warp_fields_from_curvature()
        
        n = self._synchronize_arrays()
        if n and n >= 2:
            self._interfere_causal_lines()
            self._apply_mhd_deflection_and_harvesting()
            self._apply_kenotic_love_dissipation(dt)
            self._align_phases(dt)
            self._writeback_arrays()

        self._apply_warp_bubbles()
        self._diffuse_fields()
        self._apply_force_routing(dt)
        self._synthesize_molecules()
        self._manage_cells_homeostasis()
        
        for organ in self.organs:
            organ.process(self.atoms, self.molecules)
            
        self._update_coordinates(dt)

    def _warp_fields_from_curvature(self):
        """[Causal Curvature] Mass warps the Environmental pressure field."""
        for node in self.atoms + [m for cell in self.cells for m in cell.molecules]:
            tx = int(np.clip(node.T * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(node.P * (self.size - 1) / 10.0, 0, self.size - 1))
            self.P_field[tx, px] += float(node.mass * 0.1)

    def _interfere_causal_lines(self):
        nodes = self._nodes
        n = self._n
        if n < 2: return
        
        # We need the last points of causal_lines
        last_pts = []
        valid_mask = []
        for nd in nodes:
            if len(nd.causal_line) > 0:
                last_pts.append(nd.causal_line[-1])
                valid_mask.append(True)
            else:
                last_pts.append(np.zeros(3))
                valid_mask.append(False)
        last_pts = np.array(last_pts, dtype=np.float32)
        valid_mask = np.array(valid_mask, dtype=bool)
        
        diffs = last_pts[:, np.newaxis, :] - last_pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))
        
        # Upper triangle mask, dist < 2.5, both valid
        mask = (dist < 2.5) & valid_mask[:, np.newaxis] & valid_mask[np.newaxis, :]
        mask = np.triu(mask, k=1)
        
        idx_a, idx_b = np.where(mask)
        for i, j in zip(idx_a, idx_b):
            pos_a = last_pts[i]
            pos_b = last_pts[j]
            tx = int(np.clip(((pos_a[0] + pos_b[0])/2.0) * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(((pos_a[1] + pos_b[1])/2.0) * (self.size - 1) / 10.0, 0, self.size - 1))
            self.T_field[tx, px] += 0.15
            self.P_field[tx, px] += 0.1

    def _apply_warp_bubbles(self):
        """
        [Warp Bubble: 워프 버블 시공간 왜곡 제어]
        Warp cells manipulate space: compress space in front (+P) and expand behind (-P)
        along their target trajectory, causing coordinates to glide without friction.
        """
        for cell in self.cells:
            if cell.warp_bubble_active:
                for mol in cell.molecules:
                    # Find trajectory direction towards concept alignment [5.0, 5.0, 8.0]
                    direction = np.array([5.0 - mol.T, 5.0 - mol.P, 8.0 - mol.E])
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm

                        # Compress in front of molecule: inject pressure in grid
                        front_pos = np.array([mol.T, mol.P]) + direction[:2] * 1.5
                        fx = int(np.clip(front_pos[0] * (self.size - 1) / 10.0, 0, self.size - 1))
                        fy = int(np.clip(front_pos[1] * (self.size - 1) / 10.0, 0, self.size - 1))
                        self.P_field[fx, fy] = min(10.0, self.P_field[fx, fy] + 1.2) # Compress front

                        # Expand behind molecule: reduce pressure behind
                        rear_pos = np.array([mol.T, mol.P]) - direction[:2] * 1.5
                        rx = int(np.clip(rear_pos[0] * (self.size - 1) / 10.0, 0, self.size - 1))
                        ry = int(np.clip(rear_pos[1] * (self.size - 1) / 10.0, 0, self.size - 1))
                        self.P_field[rx, ry] = max(0.1, self.P_field[rx, ry] - 0.8)  # Expand rear

                        # Zero out internal drag/friction inside the bubble
                        mol.velocity += direction * 0.15

    def _apply_mhd_deflection_and_harvesting(self):
        n = self._n
        if n < 2: return
        
        diffs = self._positions[np.newaxis, :, :] - self._positions[:, np.newaxis, :] # b - a
        dist = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-5
        mask = dist < 2.0
        np.fill_diagonal(mask, False)
        
        # vel_b (1, N, 3) x B_field_a (N, 1, 3)
        vel_b = self._velocities[np.newaxis, :, :]
        b_field_a = self._b_fields[:, np.newaxis, :]
        
        # cross product broadcast: (N, N, 3)
        cross_prod = np.cross(vel_b, b_field_a, axis=-1)
        lorentz_force = self._charges[:, np.newaxis, np.newaxis] * cross_prod
        lorentz_norm = np.sqrt(np.sum(lorentz_force**2, axis=-1))
        
        valid = mask & (lorentz_norm > 0)
        
        # Deflect velocity of B
        deflect = np.zeros_like(self._velocities)
        for i in range(n):
            for j in range(n):
                if valid[i, j]:
                    self._velocities[j] += (lorentz_force[i, j] / lorentz_norm[i, j]) * 0.12
                    
                    harvested = float(lorentz_norm[i, j] * 0.05)
                    self._harvested[i] += harvested
                    
                    target_diff = np.array([5.0, 5.0, 8.0]) - self._positions[i]
                    target_norm = np.linalg.norm(target_diff) + 1e-5
                    self._velocities[i] += (target_diff / target_norm) * harvested

    def _apply_kenotic_love_dissipation(self, dt: float):
        n = self._n
        if n < 2: return
        
        energy_diff = self._energies[:, np.newaxis] - self._energies[np.newaxis, :]
        diffs = self._positions[:, np.newaxis, :] - self._positions[np.newaxis, :, :] # a - b
        dist = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-5
        
        mask = (energy_diff > 0) & (dist < 4.0)
        np.fill_diagonal(mask, False)
        
        surrender_rate = np.where(mask, 0.15 * energy_diff / dist, 0.0)
        giving = np.minimum(surrender_rate * dt, energy_diff * 0.4)
        giving = np.where(mask, giving, 0.0)
        
        energy_given = np.sum(giving, axis=1)
        energy_received = np.sum(giving, axis=0)
        
        self._energies -= energy_given
        self._energies += energy_received
        
        pull_dir = diffs / dist[:, :, np.newaxis]
        pull_force = giving[:, :, np.newaxis] * pull_dir * 0.8
        
        self._velocities += np.sum(pull_force, axis=0)
        self._velocities[:, 2] += energy_received * 0.5

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

        self.T_field = 0.95 * self.T_field + 0.05 * 1.0
        self.P_field = 0.95 * self.P_field + 0.05 * 1.0

        self.T_field = np.clip(self.T_field, 0.1, 10.0)
        self.P_field = np.clip(self.P_field, 0.1, 10.0)

    def _align_phases(self, dt: float):
        n = self._n
        if n < 2: return
        
        diffs = self._positions[:, np.newaxis, :] - self._positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-5
        mask = dist < 4.0
        mask = np.triu(mask, k=1)
        
        diff_phase = self._phases[:, np.newaxis] - self._phases[np.newaxis, :]
        energy_exchange = np.abs(self._energies[:, np.newaxis] - self._energies[np.newaxis, :])
        
        freq_prod = self._frequencies[:, np.newaxis] * self._frequencies[np.newaxis, :]
        coupling = 0.1 * freq_prod * (1.0 + energy_exchange * 0.2) / dist
        torque = -np.where(mask, coupling * np.sin(diff_phase), 0.0)
        
        # Torque applied: a += torque / freq_a, b -= torque / freq_b
        # sum torques for a and b
        torque_on_a = np.sum(torque, axis=1)
        torque_on_b = -np.sum(torque, axis=0) # since b is axis 1 in mask
        
        total_torque = torque_on_a + torque_on_b
        
        self._phases = (self._phases + total_torque / (self._frequencies + 1e-3) * dt) % (2.0 * np.pi)

    def _apply_force_routing(self, dt: float):
        if not self.atoms: return
        wells_pos = np.array([mol.T for mol in self.molecules] + [mol.P for mol in self.molecules] + [mol.E for mol in self.molecules]).reshape(3, -1).T if self.molecules else np.empty((0, 3))
        wells_mass = np.array([mol.mass for mol in self.molecules]) if self.molecules else np.empty(0)

        for atom in self.atoms:
            if atom.is_bound: continue
            tx = int(np.clip(atom.T * (self.size - 1) / 10.0, 0, self.size - 1))
            px = int(np.clip(atom.P * (self.size - 1) / 10.0, 0, self.size - 1))
            
            grad_p = self.P_field[(tx+1)%self.size, px] - self.P_field[(tx-1)%self.size, px]
            grad_t = self.T_field[tx, (px+1)%self.size] - self.T_field[tx, (px-1)%self.size]
            
            force = np.array([-0.15 * grad_p, 0.15 * grad_t, 0.0], dtype=np.float32)
            
            if len(wells_pos) > 0:
                pos = np.array([atom.T, atom.P, atom.E])
                diff = wells_pos - pos
                dist_sq = np.sum(diff**2, axis=-1)
                dist = np.sqrt(dist_sq + 1e-3)
                mask = dist < 5.0
                if np.any(mask):
                    force_mag = 0.25 * (atom.mass * wells_mass[mask]) / (dist_sq[mask] + 0.1)
                    force += np.sum(force_mag[:, np.newaxis] * (diff[mask] / dist[mask, np.newaxis]), axis=0)
            
            atom.velocity += force * dt

        for mol in self.molecules:
            pos = np.array([mol.T, mol.P, mol.E])
            diff = np.array([5.0, 5.0, 8.0]) - pos
            dist = np.linalg.norm(diff) + 1e-5
            mol.velocity += 0.06 * (diff / dist) * dt

    def _synthesize_molecules(self):
        unbound = [a for a in self.atoms if not a.is_bound]
        n_u = len(unbound)
        if n_u < 2: return
        
        tensors = np.array([a.tensor for a in unbound])
        t_vals = np.array([a.T for a in unbound])
        p_vals = np.array([a.P for a in unbound])
        
        # Resonance: (N, D) @ (D, N) -> (N, N)
        norms = np.linalg.norm(tensors, axis=1) + 1e-9
        resonance = (tensors @ tensors.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])
        
        avg_P = (p_vals[:, np.newaxis] + p_vals[np.newaxis, :]) / 2.0
        avg_T = (t_vals[:, np.newaxis] + t_vals[np.newaxis, :]) / 2.0
        
        mask = (resonance * avg_P) > (avg_T * 0.4)
        np.fill_diagonal(mask, False)
        
        bonded_groups = []
        used = set()
        
        for i in range(n_u):
            if i in used: continue
            group = [unbound[i]]
            for j in range(i + 1, n_u):
                if j in used: continue
                if mask[i, j]:
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

    def _manage_cells_homeostasis(self):
        """Group molecules into Cells and process homeostasis."""
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
        """Update coordinates and append trajectory points (Line)."""
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

            # Record trajectory (Line)
            atom.record_causal_step()

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

            # Record trajectory (Line)
            mol.record_causal_step()

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
