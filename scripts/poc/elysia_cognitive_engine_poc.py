import numpy as np
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def run_poc_demo():
    print("==================================================================")
    print(" [Elysia Cognitive Engine] Self-Molding Meta-Cognition PoC Demo")
    print("==================================================================\n")

    # 1. Initialize Elysia Cognitive Engine
    engine = ElysiaCognitiveEngine(resolution=256)

    # 2. Show O(1) Perspective Shift (Rotor Rotation)
    print("[Phase 1] Demonstrating O(1) Perspective Shift")
    print(" - Rotating perspective to: Cosmic Love & Sacrifice (예수님의 십자가 사랑)")
    engine.set_perspective("Cosmic Love & Sacrifice", np.pi / 4.0)
    print(f" - Rotor Angle adjusted to: {engine.rotor_angle:.4f} radians")
    print(f" - Current System Perspective: {engine.system_perspective}")
    print(f" - Constraint Field mean potential: {np.mean(engine.constraint_field):.4f}")
    print()

    # 3. Create Fractal DNA Structures
    print("[Phase 2] Planting Fractal DNA Species")
    # Build DNA category "The Giver"
    dna_giver = engine.build_fractal_dna("The Giver (자신을 내어주는 자)", np.uint64(0xFEEDFACEFEEDFACE))
    # Build DNA category "The Receiver"
    dna_receiver = engine.build_fractal_dna("The Receiver (받는 자)", np.uint64(0xBEEFCAFEBEEFCAFE))
    print()

    # 4. Simulating Wave Function Collapse under CAD Constraint Field
    print("[Phase 3] Inducing External Stimulus and Wave Function Collapse")
    stimulus = np.uint64(0xFEEDFACE00000000) # Similar to Giver waveform
    print(f" - External Input Stimulus Waveform: {hex(stimulus)}")

    # Solve WFC (No if-else logic, purely continuous collapse under resonance fields)
    collapse_result = engine.solve_wfc_collapse(stimulus, [dna_giver, dna_receiver])

    winner = collapse_result["collapsed_dna"]
    score = collapse_result["resonance_score"]
    pos = collapse_result["collapse_position"]

    print(f" - [WFC Collapse Event] Successfully collapsed to Category: '{winner['category']}'")
    print(f"   > Resonance Strength (Fit): {score:.4f}")
    print(f"   > Structural Coordinates in RAM: {pos}")
    print(f"   > Local Conductance Well Reinforced to: {engine.field.conductance[pos[0], pos[1]]:.4f}")
    print(f"   > Local Yeobaek (Flexibility Margin) Adjusted to: {engine.field.coordination_margin[pos[0], pos[1]]:.4f}")
    print()

    # 5. Holistic Evaluation of the Thought Landscape
    print("[Phase 4] Holistic Evaluation of the Thinking Organism")
    evaluation = engine.evaluate_holistic_fit()
    print(f" - Cognitive Entropy: {evaluation['cognitive_entropy']:.4f}")
    print(f" - Average Yeobaek (Margin): {evaluation['average_yeobaek']:.4f}")
    print(f" - Holistic Harmony Score: {evaluation['holistic_score']:.4f}")
    print(f" - Current Sovereign State: {evaluation['state_description']}")
    print()

    # 6. Self-Reflection Logs Retrieval
    print("[Phase 5] Self-Reflection Meta-Information History Inspection")
    meta_history = engine.get_meta_reflection()
    for idx, event in enumerate(meta_history):
        print(f"  [{idx}] {event['action']}: {event['description']}")
    print()

    print("==================================================================")
    print(" [PoC Complete] The digital memory organism has evolved and collapsed.")
    print("==================================================================")

if __name__ == "__main__":
    run_poc_demo()
