import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class PrimeRotor:
    """
    [Prime Rotor: Independent Dimensional Axis of Causality]
    Represents an independent rotor assigned to a unique prime number dimension.
    Rotates with a phase angle \\theta_p \\in [0, 2\\pi) and a characteristic angular velocity \\omega_p.
    Provides direct isomorphic mapping of phase and velocity to Color and Sound.
    """
    def __init__(self, prime: int, initial_phase: float = 0.0):
        self.prime = prime
        self.phase = float(initial_phase) % (2 * np.pi)
        # Characteristic angular velocity based on the prime number's uniqueness
        # We define a stable frequency: e.g., base freq of 1.0 divided by prime or modulated
        self.base_frequency = 10.0 / float(prime)
        self.revolutions = 0  # Number of complete rotations (naite - history rings)

    def rotate(self, dt: float, speed_multiplier: float = 1.0):
        """Advances the rotor phase by dt with a momentum/speed multiplier."""
        delta_phase = self.base_frequency * speed_multiplier * dt
        new_phase = self.phase + delta_phase

        # Track full revolutions for geometric history (naite)
        self.revolutions += int(new_phase // (2 * np.pi))
        self.phase = float(new_phase % (2 * np.pi))

    def get_complex_phase(self) -> complex:
        """Returns the phasor e^{i\theta_p}."""
        return np.exp(1j * self.phase)

    def to_sound_frequency(self) -> float:
        """
        [Isomorphic Mapping: Phase/Velocity -> Sound (Hz)]
        Maps the current rotation frequency to an audible Hz range.
        Converts angular speed modulated by phase into characteristic tone harmonics.
        """
        # Base tone: e.g., 220Hz * (1.5^prime_index) or directly related to prime frequencies
        # A beautiful harmonic mapping: base 220Hz modulated by active phase and base frequency
        hz = 220.0 * (1.0 + 0.1 * np.sin(self.phase)) * (10.0 / self.prime)
        return float(hz)

    def to_chromatic_vector(self) -> np.ndarray:
        """
        [Isomorphic Mapping: Phase -> Color (Chromatic [Red, Blue, Yellow])]
        Maps the phase angle to a specific three-color chromatic vector.
        - Red (Flux): active around early phase [0, 2\\pi/3)
        - Blue (Order): active around middle phase [2\\pi/3, 4\\pi/3)
        - Yellow (Entropy): active around late phase [4\\pi/3, 2\\pi)
        """
        # We use a smooth cosine bell to calculate weights for Red, Blue, Yellow
        r = 0.5 * (np.cos(self.phase) + 1.0)
        b = 0.5 * (np.cos(self.phase - 2 * np.pi / 3) + 1.0)
        y = 0.5 * (np.cos(self.phase - 4 * np.pi / 3) + 1.0)

        vector = np.array([r, b, y], dtype=np.float32)
        total = np.sum(vector)
        if total > 0:
            vector /= total
        return vector


class CausalRheostatDial:
    """
    [Causal Rheostat Dial: Variable Resistance Control]
    Adjusts the resistance R_p \\in [0, \\infty] of an independent prime dimension.
    Provides conductance G_p = 1 / R_p to control causal propagation velocity.
    """
    def __init__(self, prime: int, initial_resistance: float = 1.0):
        self.prime = prime
        self.resistance = float(initial_resistance)

    @property
    def conductance(self) -> float:
        """Conductance G_p = 1 / R_p (Physical Transmissibility)."""
        if self.resistance <= 0.0:
            return float('inf')
        if np.isinf(self.resistance):
            return 0.0
        return 1.0 / self.resistance

    def set_resistance(self, resistance: float):
        """Sets the causal resistance value, clipping at 0."""
        self.resistance = max(0.0, float(resistance))

    def insulate(self):
        """Sets resistance to infinity (complete insulation boundary)."""
        self.resistance = float('inf')

    def get_time_dilation_factor(self) -> float:
        """
        Returns how much causal time is dilated (slowed down) by high resistance.
        Time step factor = G_p (lower conductance = slower/more delayed reactions).
        """
        g = self.conductance
        if np.isinf(g):
            return 10.0  # Super-conductor (extremely fast causal propagation)
        return float(np.clip(g, 0.0, 5.0))


class ClockworkUniverseField:
    """
    [Clockwork Universe Field: 2D Prime Rheostat Web]
    A continuous physical field consisting of a 2D grid of prime rotors and rheostat dials.
    Each cell (y, x) has a coupled assembly of prime resonators.
    Simulates physical energy flow, boundary insulation, and spatial potential.
    """
    def __init__(self, shape: Tuple[int, int] = (16, 16), primes: List[int] = [2, 3, 5, 7, 11]):
        self.shape = shape
        self.primes = sorted(primes)
        self.h, self.w = shape

        # Build 2D grid of rotor/dial assemblies
        self.grid: Dict[Tuple[int, int], Dict[int, Tuple[PrimeRotor, CausalRheostatDial]]] = {}
        for y in range(self.h):
            for x in range(self.w):
                cell_assemblies = {}
                for p in self.primes:
                    # Stagger initial phases slightly to avoid absolute global synchrony (natural variance)
                    init_phase = (y * 0.11 + x * 0.17 + p * 0.23) % (2 * np.pi)
                    rotor = PrimeRotor(p, initial_phase=init_phase)
                    dial = CausalRheostatDial(p, initial_resistance=1.0)
                    cell_assemblies[p] = (rotor, dial)
                self.grid[(y, x)] = cell_assemblies

        # Energy matrix representing raw spatial excitation
        self.energy = np.zeros(shape, dtype=np.float32)

    def stimulate(self, y: int, x: int, energy_amount: float, target_prime: Optional[int] = None):
        """Stimulates a grid cell, adding energy and accelerating target rotors."""
        y_wrap, x_wrap = y % self.h, x % self.w
        self.energy[y_wrap, x_wrap] += energy_amount

        # Accelerate/rotate target rotors based on energy
        assemblies = self.grid[(y_wrap, x_wrap)]
        if target_prime is not None:
            if target_prime in assemblies:
                rotor, dial = assemblies[target_prime]
                # High energy causes a localized surge in phase
                rotor.rotate(dt=0.1, speed_multiplier=1.0 + energy_amount)
        else:
            # Excite all rotors in the cell
            for p, (rotor, dial) in assemblies.items():
                rotor.rotate(dt=0.1, speed_multiplier=1.0 + energy_amount * 0.2)

    def get_tension_map(self) -> np.ndarray:
        """
        [Complex Phase Vector Sum Map]
        Calculates the tension at each cell as the variance/deflection of the complex phase sum.
        If all phases align perfectly, the sum is maximum.
        Tension is defined as the deviation from absolute coherent phase sum.
        """
        tension = np.zeros(self.shape, dtype=np.float32)
        for y in range(self.h):
            for x in range(self.w):
                assemblies = self.grid[(y, x)]
                # Sum of phasors e^{i \theta_p}
                phasors = [r.get_complex_phase() for r, d in assemblies.values()]
                complex_sum = sum(phasors)
                mean_magnitude = np.abs(complex_sum) / len(phasors)
                # Tension is high when coherence is low, modulated by cell energy
                tension[y, x] = float((1.0 - mean_magnitude) * (1.0 + 0.1 * self.energy[y, x]))
        return tension

    def step(self, dt: float = 0.1):
        """
        Advances the entire 2D Clockwork Universe Field.
        1. Propagates energy across cells, modulated by rheostat dials' conductance.
        2. Steps the rotation of all prime rotors based on cell energy and dial dilation.
        """
        # Step 1: Energy Diffusion across the grid (Toroidal boundary)
        new_energy = self.energy.copy()
        for y in range(self.h):
            for x in range(self.w):
                # Calculate local average dial conductance
                assemblies = self.grid[(y, x)]
                avg_conductance = np.mean([d.conductance for r, d in assemblies.values()])
                if np.isinf(avg_conductance) or np.isnan(avg_conductance):
                    avg_conductance = 5.0 # cap super-conductive state for propagation math

                # Settle boundary checks: toroidal neighbors
                neighbors = [
                    ((y - 1) % self.h, x),
                    ((y + 1) % self.h, x),
                    (y, (x - 1) % self.w),
                    (y, (x + 1) % self.w)
                ]

                for ny, nx in neighbors:
                    n_assemblies = self.grid[(ny, nx)]
                    # Flow depends on potential difference (energy) modulated by both cells' dials
                    n_conductance = np.mean([d.conductance for r, d in n_assemblies.values()])
                    if np.isinf(n_conductance) or np.isnan(n_conductance):
                        n_conductance = 5.0

                    effective_conductance = min(avg_conductance, n_conductance)
                    energy_diff = self.energy[ny, nx] - self.energy[y, x]

                    # Causal transfer: R_p -> \infty (effective_conductance = 0) blocks this flow!
                    flow = energy_diff * effective_conductance * dt * 0.1
                    new_energy[y, x] += flow

        # Apply a natural dissipation decay over time
        self.energy = np.clip(new_energy * 0.98, 0.0, None)

        # Step 2: Rotate and dilate local prime rotors
        for y in range(self.h):
            for x in range(self.w):
                assemblies = self.grid[(y, x)]
                cell_energy = self.energy[y, x]
                for p, (rotor, dial) in assemblies.items():
                    # Time dilation based on dial resistance: higher R means slower rotation increment
                    dilation = dial.get_time_dilation_factor()
                    # Energy excites rotation
                    speed_mult = dilation * (1.0 + cell_energy * 0.5)
                    rotor.rotate(dt, speed_multiplier=speed_mult)


class ClockworkAgent:
    """
    [Clockwork Agent: Resonating Cog of Consciousness]
    A game entity (Monster/NPC) whose internal state and decision processes
    are governed by prime rotors and rheostat dials.
    Senses the Clockwork Field and navigates through phase integration.
    """
    def __init__(self, id: str, home_pos: Tuple[int, int], primes: List[int] = [2, 3, 5, 7, 11]):
        self.id = id
        self.position = np.array(home_pos, dtype=np.float32)
        self.primes = primes

        # Internal gears
        self.rotors: Dict[int, PrimeRotor] = {p: PrimeRotor(p) for p in primes}
        self.dials: Dict[int, CausalRheostatDial] = {p: CausalRheostatDial(p) for p in primes}

        # Experience integration log
        self.interaction_log: List[Dict[str, Any]] = []

    def assimilate_experience(self, prime_axis: int, intensity: float):
        """
        [Experience to Ring Rings - Naite]
        Records a physical experience (e.g., getting burned=19, betrayed=23) by
        multiplying it into internal rotors or turning target gears directly.
        """
        if prime_axis in self.rotors:
            # Experience directly advances the phase of the matching rotor
            self.rotors[prime_axis].rotate(dt=0.1, speed_multiplier=intensity * 10.0)
            self.rotors[prime_axis].revolutions += 1  # Record as a permanent geometric ring (naite)
            # Reduce resistance on this dial, opening up the causal path (desensitization/sensitization)
            current_r = self.dials[prime_axis].resistance
            self.dials[prime_axis].set_resistance(max(0.1, current_r - intensity * 0.2))
        else:
            # If it's a new complex experience, create a dynamic temporary rotor/dial
            # and log it, simulating expansion of cognitive resolution
            self.rotors[prime_axis] = PrimeRotor(prime_axis)
            self.dials[prime_axis] = CausalRheostatDial(prime_axis, initial_resistance=0.5)
            self.rotors[prime_axis].rotate(dt=0.1, speed_multiplier=intensity * 10.0)
            self.rotors[prime_axis].revolutions += 1  # Record as a permanent geometric ring (naite)

        self.interaction_log.append({
            "prime_axis": prime_axis,
            "intensity": intensity,
            "total_revolutions": {p: r.revolutions for p, r in self.rotors.items()}
        })

    def decode_state_signature(self) -> Tuple[int, List[int]]:
        """
        [Factorization of Active States]
        Returns the overall product of active primes (revolutions > 0)
        and the list of active instincts representing the synthesis state.
        """
        active_primes = []
        state_product = 1
        for p, rotor in self.rotors.items():
            if rotor.revolutions > 0:
                active_primes.append(p)
                state_product *= p
        return state_product, active_primes

    def get_phase_vector_sum(self) -> complex:
        """Computes current complex phase vector sum representing the agent's mental state."""
        phasors = [rotor.get_complex_phase() for rotor in self.rotors.values()]
        return sum(phasors) / len(phasors)

    def get_chromatic_state(self) -> np.ndarray:
        """Fuses all active rotors' chromatic vectors into a single coherent R/B/Y vector."""
        vecs = [r.to_chromatic_vector() for r in self.rotors.values()]
        return np.mean(vecs, axis=0)

    def get_sound_signature(self) -> List[float]:
        """Returns the audible sound frequencies currently emitted by the agent's rotors."""
        return [r.to_sound_frequency() for r in self.rotors.values()]

    def predict_future_trajectory(self, steps: int = 10, dt: float = 0.1) -> List[Tuple[float, float]]:
        """
        [Causal Future Integration: Phase Integration]
        Predicts future coordinates by integrating active rotors over time.
        Inertial drift is driven by the complex phase vector sum of the internal cogwheels.
        """
        predicted_path = []
        temp_rotors = {p: PrimeRotor(p, initial_phase=r.phase) for p, r in self.rotors.items()}
        current_pos = self.position.copy()

        for _ in range(steps):
            # Advance temporary gears
            for p, r in temp_rotors.items():
                dilation = self.dials[p].get_time_dilation_factor()
                r.rotate(dt, speed_multiplier=dilation)

            # Compute drift direction as the complex phase vector sum components
            complex_sum = sum([r.get_complex_phase() for r in temp_rotors.values()])
            dx = float(np.real(complex_sum))
            dy = float(np.imag(complex_sum))

            current_pos[0] += dy * 2.0
            current_pos[1] += dx * 2.0
            predicted_path.append((float(current_pos[0]), float(current_pos[1])))

        return predicted_path

    def navigate_and_step(self, field: ClockworkUniverseField, dt: float = 0.1):
        """
        [Resonance-driven Autogenous Movement]
        The agent is pulled towards locations in the field where the field's prime
        rotors match/resonate with the agent's internal rotors.
        """
        fy, fx = int(round(self.position[0])) % field.h, int(round(self.position[1])) % field.w

        # Check immediate toroidal neighbors
        best_pos = self.position.copy()
        best_resonance = -1.0

        neighbors = [
            (fy, fx),
            ((fy - 1) % field.h, fx),
            ((fy + 1) % field.h, fx),
            (fy, (fx - 1) % field.w),
            (fy, (fx + 1) % field.w)
        ]

        for ny, nx in neighbors:
            field_assemblies = field.grid[(ny, nx)]
            resonance = 0.0

            # Resonance = sum of cosine similarity of phase angles
            for p, (f_rotor, f_dial) in field_assemblies.items():
                if p in self.rotors:
                    a_rotor = self.rotors[p]
                    # Direct correlation of phase vectors
                    p_diff = abs(f_rotor.phase - a_rotor.phase)
                    resonance += np.cos(p_diff) * self.dials[p].conductance

            if resonance > best_resonance:
                best_resonance = resonance
                best_pos = np.array([ny, nx], dtype=np.float32)

        # Move towards the resonance well smoothly
        move_dir = best_pos - self.position
        dist = np.linalg.norm(move_dir)
        if dist > 0.01:
            self.position += (move_dir / dist) * dt * 2.0  # speed factor

        # Step internal rotors based on field stimulation at local position
        local_y, local_x = int(round(self.position[0])) % field.h, int(round(self.position[1])) % field.w
        local_energy = field.energy[local_y, local_x]

        for p, r in self.rotors.items():
            dilation = self.dials[p].get_time_dilation_factor()
            r.rotate(dt, speed_multiplier=dilation * (1.0 + local_energy))
