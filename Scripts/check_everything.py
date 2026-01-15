import torch
from Core.Digestion.digestive_system import DigestiveSystem
from Core.Monad.monad_core import Monad
from Core.Engine.governance_engine import GovernanceEngine
from Core.Foundation.Memory.fractal_causality import CausalRole

def verify():
    print("--- 1. Monad Search ---")
    m = Monad("Elysia_Seed")
    print(f"Zero-Frequency Identity: {m.ZERO_FREQUENCY_ID}")
    print(f"Monad ID Hash: {m._id_hash}")
    
    print("\n--- 2. Causal Chain Check ---")
    elysia_mock = type('Mock', (), {'bridge': None, 'graph': None})()
    stomach = DigestiveSystem(elysia_mock)
    weights = torch.randn(5, 5)
    probe = stomach.active_probe(weights, "verification_layer")
    print(f"Active Probe (Function): {probe['function']}")
    print(f"Active Probe (Reality): {probe['reality']}")
    
    print("\n--- 3. Governance Rotor Check ---")
    gov = GovernanceEngine()
    print(f"Axiom Rotors: {list(gov.dials.keys())}")
    for name in ["Identity", "Purpose", "Future"]:
        rotor = gov.dials.get(name)
        if rotor:
            print(f"Rotor {name}: {rotor.dna.label} | DNA: {rotor.dna}")
        else:
            print(f"Rotor {name} NOT FOUND")

if __name__ == "__main__":
    verify()
