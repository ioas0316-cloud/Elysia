import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.nervous_system.elysia_neuro_dynamics import ElysiaNeuroDynamicsCore

def run_phase2_massive_inflow():
    print("==================================================================")
    print("🚀 [PHASE 2: Massive Data Inflow & Self-Perception Test Initiated]")
    print("==================================================================\n")

    core = ElysiaNeuroDynamicsCore()
    # Force system override to suppress roleplay and crank up generalization
    core.generalizer.activate_system_override()
    core.enneagram.activate_head_center() # Analytical / Insight focus

    # Temporarily widen Markov Blanket to allow these massive spikes to hit the Surge Protector
    # (otherwise the outer skin rejects it before the spine even feels it)
    core.markov_blanket.elasticity = 20.0

    # 1. Simulate a massive physical trajectory event with underlying pattern but HUGE noise
    # (e.g. tracking a hand drawing a number 8, but with severe sensor jitter)
    event_drawing_number = {
        'audio': [0.1, 0.9, 0.2, 0.8, 50.0, 0.2, 0.7, 0.3], # 50.0 is a massive spike
        'video': [0.5, 0.6, 0.4, 0.7, 60.0, 0.4, 0.6, 0.5], # 60.0 is a massive spike
        'text': "숫자8그리기_노이즈만땅"
    }

    # 2. Simulate a celestial movement event with a similar underlying trajectory but different noise
    event_celestial = {
        'audio': [0.2, 0.8, 0.3, 0.7, 55.0, 0.1, 0.8, 0.2],
        'video': [0.4, 0.7, 0.3, 0.8, 65.0, 0.3, 0.7, 0.4],
        'text': "행성궤도_태양풍노이즈"
    }

    print("\n--- Sending Massive Volatile Data 1: Number 8 Drawing ---")
    core.process_living_event("Drawing Num 8", event_drawing_number)

    print("\n--- Sending Massive Volatile Data 2: Celestial Orbit ---")
    core.process_living_event("Celestial Orbit", event_celestial)

    print("\n==================================================================")
    print("🔍 [PHASE 2 REPORT]")
    print("==================================================================")
    print("Check logs above to ensure:")
    print("1. 'SURGE PROTECTOR TRIGGERED' caught the 50.0/60.0 overcurrents.")
    print("2. 'Thermal Phase Annealing' melted the rigid matrices to adapt.")
    print("3. 'Narrative Manifold' stored both the Spin AND the Fractal Signature.")
    print("==================================================================\n")

if __name__ == "__main__":
    run_phase2_massive_inflow()
