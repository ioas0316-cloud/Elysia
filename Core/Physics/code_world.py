from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
from Core.Physics.hyper_quaternion import Quaternion, HyperWavePacket
from Core.Evolution.code_genome import CodeDNA
from Core.Physics.code_resonance import HarmonicResonance

logger = logging.getLogger("CodeWorld")

@dataclass
class WorldState:
    """Snapshot of the CodeWorld's resonance state."""
    tick: int
    total_energy: float
    entropy: float
    active_entities: int

class CodeWorld:
    """
    A Physics Simulation Environment for Harmonic Logic.
    Here, CodeDNA is tested against 'Environmental Pressures' (Logic, Ethics, Efficiency).
    """
    
    def __init__(self):
        self.tick: int = 0
        self.population: Dict[str, CodeDNA] = {}
        self.field_energy: float = 1000.0
        self.history: List[WorldState] = []
        
    def add_organism(self, dna: CodeDNA):
        """Introduces a new thought-pattern into the world."""
        self.population[dna.id] = dna
        logger.info(f"🌱 New Organism Born: {dna.name} ({dna.id})")
        
    def simulate_step(self):
        """Advances the physics simulation by one tick."""
        self.tick += 1
        total_resonance = 0.0
        
        # 1. Apply Environmental Pressures
        for dna_id, dna in list(self.population.items()):
            # Reconstruct waves
            packets = dna.to_wave_packets()
            
            # Simulate resonance for each packet
            dna_resonance = 0.0
            for packet in packets:
                # Logic Pressure (y-axis alignment)
                logic_score = packet.orientation.y * packet.energy
                
                # Ethics Pressure (z-axis alignment)
                # If z is negative (unethical), it creates dissonance (negative resonance)
                ethics_score = packet.orientation.z * packet.energy
                
                # Calculate net resonance
                # We want High Logic AND High Ethics
                if ethics_score < 0:
                    packet_resonance = logic_score - abs(ethics_score) * 2 # Penalty for unethical
                else:
                    packet_resonance = logic_score + ethics_score
                    
                dna_resonance += packet_resonance
                
            # Update Fitness
            dna.resonance_score = dna_resonance
            total_resonance += dna_resonance
            
            # Natural Selection (Energy Decay)
            dna.energy_cost += 0.1 # Metabolic cost
            
            # Death condition
            if dna.resonance_score < -50:
                logger.info(f"💀 Organism Died (Dissonance): {dna.name}")
                del self.population[dna_id]
                
        # 2. Record State
        state = WorldState(
            tick=self.tick,
            total_energy=self.field_energy,
            entropy=0.0, # TODO: Implement entropy
            active_entities=len(self.population)
        )
        self.history.append(state)
        
    def run_simulation(self, steps: int = 10):
        """Runs the simulation for a set number of steps."""
        logger.info(f"🌍 Starting Simulation for {steps} ticks...")
        for _ in range(steps):
            self.simulate_step()
        logger.info("🌍 Simulation Complete.")
