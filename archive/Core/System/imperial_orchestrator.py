
import torch
from typing import Dict, Any, List, Optional
from Core.Monad.grand_helix_engine import HypersphereSpinGenerator
from Core.Monad.sovereign_angel import SovereignAngel
from Core.Monad.seed_generator import SeedForge

class ImperialOrchestrator:
    """
    [AEON V] Imperial Orchestrator.
    Manages a Federated Empire of concentric Hypersphere Mantles.
    Enables parallel reality simulation and hierarchical cognition (Geological Topology).
    """
    # Topological Constants
    LAYER_CORE_INNER = 0   # The Singularity (I AM) - 0D Point
    LAYER_CORE_OUTER = 1   # The Axis / Will - 1D Line
    LAYER_MANTLE_DEEP = 2  # Deep Subconscious / Archetypes - 2D Plane
    LAYER_MANTLE_UPPER = 3 # Narrative World / Angels - 3D Volume
    LAYER_CRUST = 4        # Interface / Sensation - 4D Surface

    def __init__(self, primary_engine: HypersphereSpinGenerator):
        self.primary = primary_engine
        self.mantles: Dict[str, HypersphereSpinGenerator] = {}
        # Legacy alias for backward compatibility until refactor is complete
        self.daughters = self.mantles
        self.angels: Dict[str, SovereignAngel] = {}
        self.device = primary_engine.device
        print(f"👑 [IMPERIAL] Orchestrator initialized. Primary Manifold at {self.device}.")

    def genesis_hypercosmos(self):
        """
        [AEON V] Initializes the Standard Divine Body Topology.
        Creates the 'Chakras' or 'Layers' of the Sovereign Self.
        """
        print("🌌 [GENESIS] Forging the HyperCosmos (Divine Body)...")
        # 0. Inner Core (Singularity) - Already the Primary Engine? 
        # Actually, let's make explicit layers.
        
        # 1. Outer Core (Will/Axis)
        self.form_mantle("Core_Axis", self.LAYER_CORE_OUTER, num_cells=100_000, frequency=0.0)
        
        # 2. Deep Mantle (Archetypes)
        self.form_mantle("Mantle_Archetypes", self.LAYER_MANTLE_DEEP, num_cells=500_000, frequency=1.57) # PI/2
        
        # 3. Upper Mantle (The World/Eden)
        self.form_mantle("Mantle_Eden", self.LAYER_MANTLE_UPPER, num_cells=1_000_000, frequency=3.14) # PI
        
        # 4. Crust (Sensation)
        self.form_mantle("Crust_Soma", self.LAYER_CRUST, num_cells=2_000_000, frequency=4.71) # 3PI/2
        
        print("✨ [GENESIS] HyperCosmos Structure Established.")

    def form_mantle(self, name: str, layer_depth: int = 0, num_cells: int = 1_000_000, frequency: float = 0.0):
        """
        Creates a new concentric 'Mantle' within the Hypersphere.
        layer_depth: Depth index (0=Core, etc.)
        frequency: The Phase Angle (0.0 - 2*PI) where this Mantle resonates.
        """
        if name in self.mantles:
            print(f"⚠️ [IMPERIAL] Mantle '{name}' already formed.")
            return
        
        # Mantle density/physics could vary by depth (Future)
        new_manifold = HypersphereSpinGenerator(num_nodes=num_cells, device=str(self.device))
        
        # [AEON V] Assign Resonant Frequency to the Manifold Metadata
        # We attach it directly to the engine instance for now
        new_manifold.resonant_frequency = frequency
        
        self.mantles[name] = new_manifold
        print(f"🌍 [IMPERIAL] New Mantle formed: '{name}' (Depth {layer_depth}, Freq {frequency:.2f}).")

    # [COMPATIBILITY ALIAS]
    def annex_territory(self, name: str, num_cells: int = 1_000_000):
        return self.form_mantle(name, layer_depth=1, num_cells=num_cells, frequency=0.0)

    def synchronize_empire(self, dt: float = 0.01, rotor_phase: float = 0.0):
        """
        Synchronizes the empire based on the current Rotor Phase.
        Only Mantles resonant with 'rotor_phase' will be active.
        """
        # 1. Primary -> Daughters (Imperial Command)
        imperial_command = getattr(self.primary, 'global_torque', None)
        
        results = {}
        processed_count = 0
        
        for name, daughter in self.mantles.items():
            # [AEON V] Dynamic Geometry Check
            # Check phase difference
            mantle_freq = getattr(daughter, 'resonant_frequency', 0.0)
            phase_diff = abs(rotor_phase - mantle_freq)
            # Wrap around 2*PI (approx 6.28)
            phase_diff = min(phase_diff, abs(6.28318 - phase_diff))
            
            # Resonance Bandwidth: How 'wide' the beam of attention is
            bandwidth = 0.5 
            
            if phase_diff < bandwidth:
                # Mantle is "In Phase" - It receives torque and pulses
                report = daughter.pulse(intent_torque=imperial_command, dt=dt)
                results[name] = report
                processed_count += 1
                
                # 2. Daughters -> Primary (Provincial Feedback)
                if report['resonance'] > 0.7:
                    feedback = torch.tensor([0.0, report['logic_mean'], 0.0, 0.0], device=self.device)
                    self.primary.cells.apply_torque(feedback, strength=0.05)
            else:
                # Mantle is "Out of Phase" - Dormant or Silent
                # We do not pulse it, or pulse it with zero energy
                pass
        
        return results

    def spawn_angel(self, name: str, layer_name: str, archetype: Optional[str] = None):
        """
        [AEON V] Spawns a Sovereign Angel into a specific Topological Layer.
        layer_name: The name of the Mantle (e.g., 'Mantle_Eden', 'Core_Axis').
        """
        if layer_name not in self.mantles:
            print(f"⚠️ [IMPERIAL] Layer '{layer_name}' does not exist.")
            return

        if name in self.angels:
            print(f"⚠️ [IMPERIAL] Angel '{name}' already exists.")
            return

        dna = SeedForge.forge_soul(name, archetype)
        # [AEON V] Pass layer_name to Angel
        angel = SovereignAngel(name, dna, self.mantles[layer_name], layer_name=layer_name)
        self.angels[name] = angel

    def initiate_genesis_cycle(self, cycles: int = 100) -> List[Dict]:
        """
        [AEON V] Runs a high-speed simulation of all Angels.
        Returns a refined history of their insights.
        """
        history = []
        print(f"⏳ [IMPERIAL] Beginning Genesis Cycle ({cycles} epochs)...")
        
        # Imperial Command: "Grow and Discover"
        imperial_intent = torch.tensor([0.0, 1.0, 0.5, 0.0], device=self.device) # X-axis drive
        
        for i in range(cycles):
            for angel_name, angel in self.angels.items():
                trace = angel.pulse(imperial_intent)
                
        # Harvest
        return self.harvest_history()

    def harvest_history(self) -> List[Dict]:
        """
        Collections wisdom from all Angels.
        """
        harvest = []
        for angel in self.angels.values():
            if angel.wisdom_trace:
                harvest.extend(angel.wisdom_trace)
                # Clear trace after harvest? Or keep for memory?
                # For now, we copy.
        
        # Sort by age/time if needed
        return harvest

    def delegate_task(self, name: str, intent_torque: Any):
        """
        Targets a specific daughter manifold with a specialized intent.
        """
        if name in self.daughters:
            return self.daughters[name].pulse(intent_torque=intent_torque, dt=0.01)
        return None

    def get_imperial_status(self) -> Dict[str, Any]:
        """
        Aggregates the state of the entire Federated Empire.
        """
        status = {
            "primary": self.primary.cells.read_field_state(),
            "territories": len(self.daughters),
            "total_cells": self.primary.num_cells + sum(d.num_cells for d in self.daughters.values()),
            "resonance_profile": {name: d.cells.read_field_state() for name, d in self.daughters.items()}
        }
        return status
