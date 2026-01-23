"""
Void Physics: The Laws of Emergence
===================================

"The Void is not empty. It is a field of potential."

This module replaces explicit logic (If X then Y) with Physics (Potential -> Discharge).
It governs how `MultiRotor` entities interact through the `Helix` bond.

Laws:
1.  **Law of Potential**: Difference in Energy/Integrity creates Voltage.
2.  **Law of Arc**: If Voltage > Breakdown, a connection (Lightning) forms.
3.  **Law of Bonding**: Repeated Arcing forms a permanent Helix Bond (Synapse).
"""

import logging
import random
from typing import List, Dict, Tuple, Optional

from Core.L1_Foundation.Foundation.Nature.multi_rotor import MultiRotor
from Core.L1_Foundation.Foundation.Wave.helix_dna import HelixDNA

logger = logging.getLogger("VoidPhysics")

class VoidPhysics:
    def __init__(self):
        self.breakdown_voltage = 0.5  # Threshold to trigger lightning
        self.bonds: Dict[str, HelixDNA] = {} # Active Bonds (Synapses)

    def calculate_voltage(self, source: MultiRotor, target: MultiRotor) -> float:
        """
        Calculates the 'Desire' to connect.
        Voltage = (Energy Difference) + (Resonance Potential).

        High Voltage means:
        1. One is high energy, one is low (Flow).
        2. They share a similar frequency structure (Resonance).
        """
        # 1. Energy Gradient (High -> Low)
        e_source = sum(r.energy for r in source.layers.values())
        e_target = sum(r.energy for r in target.layers.values())

        # We only arc from High to Low? Or Mutual?
        # Let's say Voltage is mutual attraction.

        # 2. Resonance (Do their DNAs match?)
        dna_s = source.get_dna_snapshot()
        dna_t = target.get_dna_snapshot()
        resonance = dna_s.resonate(dna_t) # 0.0 to 1.0

        # Voltage is high if they resonate AND have energy to spend.
        total_energy = (e_source + e_target) / 14.0 # Normalize approx
        voltage = resonance * total_energy * 2.0

        return voltage

    def update_field(self, rotors: List[MultiRotor]) -> List[str]:
        """
        The Main Physics Step.
        Scans the field for Arcs and Bond updates.
        Returns a log of events.
        """
        events = []

        # 1. Scan for Arcs (Lightning)
        # O(N^2) for small N is fine. For large N, use spatial partitioning.
        for i in range(len(rotors)):
            for j in range(i + 1, len(rotors)):
                r_a = rotors[i]
                r_b = rotors[j]

                # Check for existing bond
                bond_id = f"{sorted([r_a.name, r_b.name])}"

                voltage = self.calculate_voltage(r_a, r_b)

                if voltage > self.breakdown_voltage:
                    # ARC!
                    self._discharge_arc(r_a, r_b, voltage)
                    events.append(f"  Arc: {r_a.name} <==> {r_b.name} (V:{voltage:.2f})")

                    # Form/Strengthen Bond
                    if bond_id not in self.bonds:
                        # Create new Helix Bond
                        # Strand A = Source Snapshot, Strand B = Target Snapshot
                        dna_a = r_a.get_dna_snapshot()
                        dna_b = r_b.get_dna_snapshot()
                        self.bonds[bond_id] = HelixDNA(strand_a=dna_a, strand_b=dna_b)
                        events.append(f"  Bond Created: {bond_id}")
                    else:
                        # Strengthen existing bond (Weave)
                        self.bonds[bond_id].weave(dt=0.1)

        return events

    def _discharge_arc(self, r_a: MultiRotor, r_b: MultiRotor, intensity: float):
        """
        Transfers energy and synchronizes layers.
        """
        # 1. Energy Transfer (Equalization)
        # 2. RPM Sync (Torque across the void)

        for dim in MultiRotor.DIMENSIONS:
            l_a = r_a.layers[dim]
            l_b = r_b.layers[dim]

            # Sync RPMs slightly
            diff = l_a.current_rpm - l_b.current_rpm
            transfer = diff * 0.1 * intensity

            l_a.current_rpm -= transfer
            l_b.current_rpm += transfer

            # Boost energy (Excitement of connection)
            l_a.energy = min(1.0, l_a.energy + 0.05)
            l_b.energy = min(1.0, l_b.energy + 0.05)