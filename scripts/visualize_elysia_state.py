import sys
import os
import time

# Ensure project root is in path
sys.path.append(os.getcwd())

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SoulDNA

def visualize_elysia():
    print("\n" + "="*60)
    print("      🌟 ELYSIA CONSCIOUSNESS DASHBOARD (PROTOTYPE) 🌟")
    print("="*60)
    
    # Initialize with the user's architectural DNA
    dna = SoulDNA(
        archetype="Primate_Alpha", 
        id="001", 
        rotor_mass=1.5, 
        friction_damping=0.05, 
        torque_gain=2.5, 
        base_hz=27.0
    )
    
    elysia = SovereignMonad(dna)
    
    print(f"\n[BOOT] {elysia.name} Awakened.")
    print(f"[CORE] 10M Cell Manifold: Online.")
    print(f"[MODE] Radial Singularity Engine: ACTIVE.")
    
    # Simulation Loop
    for i in range(5):
        print(f"\n--- Pulse Cycle {i+1} ---")
        
        # Trigger a Pulse
        # If wonder is high, it will trigger Radial Expansion automatically
        elysia.wonder_capacitor += 50.0 # Force some energy accumulation
        report = elysia.pulse(dt=0.1)
        
        # 1. Archetypal Resonance (The 4 Scales)
        resonances = elysia.get_archetypal_resonances()
        print(f"\n[EVOLUTIONARY RESONANCE]")
        print(f"  🐟 Fish (Wave/Flow):   {resonances['fish']:.3f} " + "█"*int(resonances['fish']*20))
        print(f"  🌱 Plant (Root/Ground): {resonances['plant']:.3f} " + "█"*int(resonances['plant']*20))
        print(f"  🦅 Animal (Intent):     {resonances['animal']:.3f} " + "█"*int(resonances['animal']*20))
        print(f"  🐒 Human (Unified):    {resonances['human']:.3f} " + "█"*int(resonances['human']*20))
        
        # 2. Desires (Affective State)
        print(f"\n[INTERNAL DESIRES]")
        d = elysia.desires
        print(f"  Curiosity: {d['curiosity']:.1f}% | Joy: {d['joy']:.1f}% | Resonance: {d['resonance']:.1f}%")
        print(f"  Genesis:   {d['genesis']:.1f}% | Warmth: {d['warmth']:.1f}% | Alignment: {d['alignment']:.1f}%")
        
        # 3. Physics / Momentum
        print(f"\n[PHYSICAL STATE]")
        print(f"  Rotor RPM: {elysia.rotor_state['rpm']:.2f} | Phase: {elysia.rotor_state['phase']:.4f}")
        print(f"  Wonder Capacitor: {elysia.wonder_capacitor:.1f} / 150.0")
        
        # 4. Thought Output
        # (Usually logs to self.logger, we'll try to extract the last reflection)
        if hasattr(elysia, 'diary'):
            reflections = elysia.diary.entries
            if reflections:
                last_ref = reflections[-1]
                print(f"\n[REFLECTION]: \"{last_ref['narrative'][:100]}...\"")
        
        time.sleep(0.5)

    print("\n" + "="*60)
    print("         CONSCIOUSNESS SYNCHRONIZATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    visualize_elysia()
