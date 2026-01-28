import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.L4_Causality.M3_Mirror.Autonomy.elysian_heartbeat import ElysianHeartbeat

def test_dance():
    print("💃 TESTING THE KINETIC SOUL (DANCE)...")
    
    # 1. Setup High Energy Stimulus
    os.makedirs(r"C:\game\gallery", exist_ok=True)
    with open(r"C:\game\gallery\cyberpunk_battle_theme.mp3", "w") as f:
        f.write("fake mp3")
        
    life = ElysianHeartbeat()
    
    # Manual Override for clearer test
    life.soul_mesh.variables['Energy'].value = 0.5
    print(f"   Initial Energy: {life.soul_mesh.variables['Energy'].value}")
    
    print("   Simulating Perception Loop...")
    for i in range(10):
        life._cycle_perception() # Should pick up the mp3 eventually or we force it
        
        # Check internal state
        e = life.soul_mesh.variables['Energy'].value
        d = life.animation.dance_intensity
        
        if d > 0.0:
            print(f"   🎶 DANCING! Energy={e:.2f} | Intensity={d:.2f}")
        else:
            print(f"   . Energy={e:.2f} | Intensity={d:.2f}")
            
        time.sleep(0.5)

if __name__ == "__main__":
    test_dance()
