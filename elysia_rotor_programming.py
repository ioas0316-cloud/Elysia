# -*- coding: utf-8 -*-
"""
ELYSIA ROTOR-ORIENTED PROGRAMMING FRAMEWORK v0.2
"The paradigm shift: Variables are Rotors. Operations are Physics. GC is Natural Decay."

This framework turns conventional variables into living Fractal Rotors.
- Variables have Energy, Phase, Scale, and Age.
- Operations (+, *) cause Physical Interactions (Attraction/Repulsion, Phase Sync, Wave Interference).
- Transients naturally decay back to the Neutral Point.
"""

import numpy as np
import time
import cmath

class Rotor:
    """
    A living variable in the Rotor Universe.
    """
    def __init__(self, name, universe, value=0.0, lock_config="1111", is_transient=False):
        self.name = name
        self.universe = universe
        self.energy = float(abs(value)) if isinstance(value, (int, float)) else 1.0

        # Determine initial phase based on value polarity to create tension
        if isinstance(value, (int, float)) and value < 0:
            self.phase = np.random.uniform(np.pi, 2*np.pi)
        else:
            self.phase = np.random.uniform(0, np.pi)

        self.scale = 1.0
        self.lock_config = lock_config
        self.age = 0.0
        self.is_transient = is_transient
        self.parent = None
        self.apply_lock(lock_config)

    def apply_lock(self, config):
        """1000~1111 configuration applies geometric constraints to the variable."""
        self.lock_config = config
        active = sum(int(b) for b in config)
        self.scale = max(0.1, (active / 4.0) * 1.8)

    def observe(self):
        """Observation forces the wave to collapse into a classical value."""
        if sum(int(b) for b in self.lock_config) == 0:
            return 0.0 # Fully locked dimension

        # The actual value is a projection of energy and phase
        observed_value = self.energy * self.scale * np.cos(self.phase)
        return float(observed_value)

    def age_tick(self, dt=0.1):
        """Natural progression of time, causing decay or phase shifts."""
        self.age += dt

        # Only transients naturally decay unless forced.
        # Persistent variables hold their energy but their phase drifts.
        if self.is_transient:
            self.energy *= np.exp(-0.15 * dt)

        # Wave-driven phase progression
        self.phase += (0.3 * self.energy * dt) % (2 * np.pi)

    # --- Magic Methods to Hook Python Syntax ---

    def __add__(self, other):
        """Attraction and Repulsion (Interference)"""
        if isinstance(other, (int, float)):
            other = self.universe.declare(f"_temp_{id(other)}", value=other, is_transient=True)

        # Physics: constructive or destructive interference based on phase difference
        phase_diff = abs(self.phase - other.phase)
        force = (self.energy * other.energy) / (phase_diff + 0.1)

        result_energy = self.energy + other.energy + force * 0.2
        result_phase = (self.phase + other.phase) / 2.0

        # Spawn a new transient rotor
        result_name = f"_({self.name}+{other.name})"
        new_rotor = self.universe.declare(result_name, value=0.0, is_transient=True)
        new_rotor.energy = result_energy
        new_rotor.phase = result_phase
        new_rotor.parent = self

        self.universe.contribute_to_neutral(result_energy * 0.1, result_phase)
        print(f"  ⚡ [Torque: Add] {self.name} & {other.name} -> {result_name} (Interference)")
        return new_rotor

    def __sub__(self, other):
        """Phase-shifted interference (Anti-resonance)"""
        if isinstance(other, (int, float)):
            other = self.universe.declare(f"_temp_{id(other)}", value=other, is_transient=True)

        phase_diff = abs(self.phase - (other.phase + np.pi)) # Inverted phase
        force = (self.energy * other.energy) / (phase_diff + 0.1)

        result_energy = abs(self.energy - other.energy) + force * 0.1
        result_phase = self.phase + np.pi/4

        result_name = f"_({self.name}-{other.name})"
        new_rotor = self.universe.declare(result_name, value=0.0, is_transient=True)
        new_rotor.energy = result_energy
        new_rotor.phase = result_phase

        print(f"  ⚡ [Torque: Sub] {self.name} & {other.name} -> {result_name} (Anti-resonance)")
        return new_rotor

    def __mul__(self, other):
        """Phase Synchronization & Energy Amplification"""
        if isinstance(other, (int, float)):
            other = self.universe.declare(f"_temp_{id(other)}", value=other, is_transient=True)

        # Physics: synchronization
        phase_sync = np.cos(self.phase - other.phase)
        result_energy = self.energy * other.energy * (1.0 + abs(phase_sync) * 0.8)

        # They pull each other's phases closer
        result_phase = (self.phase * self.energy + other.phase * other.energy) / (self.energy + other.energy + 1e-8)

        result_name = f"_({self.name}*{other.name})"
        new_rotor = self.universe.declare(result_name, value=0.0, is_transient=True)
        new_rotor.energy = result_energy
        new_rotor.phase = result_phase
        new_rotor.parent = self

        self.universe.contribute_to_neutral(result_energy * 0.2, result_phase)
        print(f"  🌪️ [Torque: Mul] {self.name} & {other.name} -> {result_name} (Phase Sync)")
        return new_rotor

    def __str__(self):
        return f"<Rotor '{self.name}': Obs={self.observe():.3f}, E={self.energy:.2f}, Ph={self.phase:.2f}>"

    def __repr__(self):
        return self.__str__()


class RotorUniverse:
    """
    The fabric of space where Rotors exist, interact, and decay.
    """
    def __init__(self):
        self.rotors = {}
        self.neutral_point = 0.0 + 0j
        self.global_time = 0.0

    def declare(self, name, value=0.0, lock_config="1111", is_transient=False):
        """Weaving a new Rotor into the universe."""
        if name in self.rotors:
            # Overwriting a variable means overriding its energy
            self.rotors[name].energy = abs(value)
            return self.rotors[name]

        rotor = Rotor(name, self, value, lock_config, is_transient)
        self.rotors[name] = rotor
        if not is_transient:
            print(f"[{name}] 결선 완료 | Energy={rotor.energy:.3f} | Lock={lock_config}")
        return rotor

    def contribute_to_neutral(self, energy, phase):
        """Delta-Y Convergence feedback to the universal neutral point."""
        self.neutral_point += cmath.rect(energy, phase)

    def evolve(self, steps=1, dt=0.1):
        """Advance time, causing decay and natural garbage collection."""
        for _ in range(steps):
            self.global_time += dt
            to_remove = []

            for name, rotor in self.rotors.items():
                rotor.age_tick(dt)

                # Natural Garbage Collection
                if rotor.is_transient and rotor.energy < 0.05:
                    to_remove.append(name)

            for name in to_remove:
                print(f"  ⚰️ [{name}] 자연 수렴 -> 중성점으로 귀환 (GC)")
                del self.rotors[name]

    def status(self):
        """Display the state of the universe."""
        print(f"\n🌌 [Universe Status] Time: {self.global_time:.1f} | Neutral Point: {abs(self.neutral_point):.4f}")
        for name, rotor in self.rotors.items():
            print(f"  {rotor}")
        print("-" * 50)


# ==================== DEMONSTRATION ====================
if __name__ == "__main__":
    print("==========================================================")
    print("🚀 ELYSIA ROTOR-ORIENTED PROGRAMMING POC v0.2")
    print("==========================================================\n")

    univ = RotorUniverse()

    # 1. Variable Declaration (Rotor Binding)
    print("--- 1. Declaration ---")
    x = univ.declare("x", 5.0, "1111")
    y = univ.declare("y", 3.0, "1011")

    # 2. Operations (Physical Interactions)
    print("\n--- 2. Operations (Torque & Interference) ---")
    # This evaluates: x + y
    z = x + y

    # This evaluates: (x + y) * 2
    w = z * 2

    # This evaluates: w - y
    final = w - y

    # 3. Observation
    print("\n--- 3. Observation (Wave Collapse) ---")
    print(f"Observed x     : {x.observe():.3f}")
    print(f"Observed final : {final.observe():.3f}")

    univ.status()

    # 4. Natural Decay (Garbage Collection via time)
    print("\n--- 4. Time Evolution (Natural GC) ---")
    print("Evolution spanning 30 ticks...")
    univ.evolve(steps=30, dt=0.2)

    univ.status()
    print("Notice how transient variables (starting with '_') decayed and returned to the neutral point.")
    print("The memory is self-healing.")
