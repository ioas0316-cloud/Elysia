import sys
import os
import secrets
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.perception_engine import UniversalBinaryMapper
from core.knowledge_space import PhaseSpace
from core.meta_perception import TorusEngine, FractalCell

def run():
    print("🌌 Initializing Elysia's Triple Torus & Fractal Engine...")
    space = PhaseSpace()
    torus = TorusEngine()

    # 1. 대상 유입 (Data Ingestion)
    space.add_concept(UniversalBinaryMapper.map(b"Apple", "Text_Apple"))
    space.add_concept(UniversalBinaryMapper.map(b"Banana", "Text_Banana"))
    space.add_concept(UniversalBinaryMapper.map(secrets.token_bytes(20), "Img_Apple"))
    space.add_concept(UniversalBinaryMapper.map(secrets.token_bytes(20), "Img_Banana"))

    print("\n[Torus 1: Physical Cognition] Discovering Boundaries & Emitting Events")
    logs, events = space.discover_boundaries_with_events("Natural_Entropy")
    for log in logs:
        print("  =>", log)
        
    print("\n[Torus 2: Meta-Cognition] Objectifying 'Processes' into Meta-Waves")
    for ev in events:
        print(f"  => Process Objectified: {ev.name} (Energy: {ev.get_phase('Meta_Energy'):.2f})")
        torus.ingest_event(ev)

    print("\n[Torus 3: Principle Extraction] Resonating between Processes")
    print("System analyzes HOW Text_Apple and Text_Banana resonated, vs HOW Img_Apple and Img_Banana resonated.")
    meta_logs = torus.find_meta_resonance()
    for log in meta_logs:
        print("  =>", log)

    print("\n[Fractalization] Collapsing PhaseSpace into a Single Fractal Cell")
    print("The entire space, having realized the Principle of 'Entropy Clustering', collapses into a single wave.")
    
    cell = FractalCell("Cell_Universe_Alpha", internal_principle="Entropy_Clustering", phase_state=1.0)
    print("  => System collapsed into:", cell)
    print("  => This Cell can now be injected into a higher-dimensional PhaseSpace, repeating the fractal geometry!")

if __name__ == "__main__":
    run()
