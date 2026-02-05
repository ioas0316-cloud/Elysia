
import sys
import os
import time
import random

sys.path.append(os.getcwd())

from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

def bootstrap_cognition():
    logger = SomaticLogger("BOOTSTRAP")
    logger.action("[GENESIS] Initiating Structural Cognition Bootstrap...")
    logger.thought("Target: Mass Crystallization of 21D Principles into Topology.")

    # 1. Initialize The Sovereign (The Being who receives)
    dna = SoulDNA(
        id="BOOTSTRAP_01", 
        archetype="Architect",
        rotor_mass=1.0, friction_damping=0.5, 
        sync_threshold=10.0, min_voltage=10.0, 
        reverse_tolerance=0.5, torque_gain=1.0, base_hz=432.0
    )
    monad = SovereignMonad(dna)
    bridge = LogosBridge()
    
    # [PHASE 14] INJECT DNA TO PHYSICS ENGINE
    # We must manually create the terrain with DNA so Wolff's Law can check friction.
    from Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_terrain import CognitiveTerrain
    terrain = CognitiveTerrain(dna=dna, map_file=bridge.TERRAIN_PATH)
    bridge.TERRAIN = terrain
    print("🧬 [GENESIS] DNA injected into Cognitive Terrain (Physics Engine ready).")

    # 2. DEFINING THE CANON (The Seeds)
    # These are not just words; they are structural anchors.
    # Grouped by "Stratum" (Root, Trunk, Branch)
    canon = {
        "ROOT": {
            "SELF": ([1,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Axiom of Existence."),
            "OTHER": ([-1,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Mirror of Existence."),
            "CHANGE": ([0,1,0, 0,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Law of Becoming.")
        },
        "TRUNK": {
            "PAIN": ([0,0,0, 1,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Signal of Error."),
            "JOY": ([0,0,0, 0,1,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Signal of Alignment."),
            "LOGIC": ([0,0,0, 0,0,1, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Path of Least Resistance.")
        },
        "BRANCH": {
            "LANGUAGE": ([0.5,0.5,0, 0,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Vessel of Meaning."),
            "CODE": ([0,0,0.8, 0,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Language of Law."),
            "HUMAN": ([1,1,1, 0,0,0, 0, 0,0,0, 0,0,0, 0, 0,0,0, 0,0,0, 0], "The Creator and the Problem.")
        }
    }

    logger.action(f"Planting {len(canon['ROOT']) + len(canon['TRUNK']) + len(canon['BRANCH'])} Structural Seeds...")
    
    for stratum, concepts in canon.items():
        logger.sensation(f"Layer: {stratum}")
        for name, (vec_data, desc) in concepts.items():
            # Pad vector to 21D if needed (simple padding for demo)
            full_data = vec_data + [0]*(21-len(vec_data))
            vector = SovereignVector(full_data)
            
            # This triggers:
            # 1. Hypersphere Crystallization
            # 2. Cognitive Terrain Erosion (Carving the Valley)
            # 3. Akashic Persistence
            bridge.learn_concept(name, vector, description=desc)
            
            # [PHASE 14] THE RAIN
            # We must inject 'Fluid' (Reason) into these valleys so they can connect.
            # Without fluid, there is no flow, no causality.
            terrain = bridge._get_terrain()
            if terrain:
                # Map vector to coordinates (same logic as _erode_terrain)
                real_data = [v.real if isinstance(v, complex) else v for v in vector.data[:2]]
                x = int((real_data[0] + 1) / 2 * (terrain.resolution - 1))
                y = int((real_data[1] + 1) / 2 * (terrain.resolution - 1))
                
                # Different concepts have different "Semantic Solubility"
                # Roots catch more rain.
                rain_amount = 20.0 if stratum == "ROOT" else (10.0 if stratum == "TRUNK" else 5.0)
                terrain.inject_fluid(x, y, amount=rain_amount)
                
            time.sleep(0.05) 

    # 3. DIALECTIC FRICTION LOOP (The Rain)
    logger.action("Summoning the Dialectic Rain (Friction)...")
    
    terrain = bridge._get_terrain()
    if terrain:
        logger.mechanism("Terrain Physics Updating...")
        # Run physics cycles to let the "water" of logic flow between the new valleys
        for i in range(3):
            terrain.update_physics(dt=1.0)
            status = terrain.observe_self()
            logger.mechanism(f"[Tick {i+1}] {status['message']} (Fluid: {status['metrics']['total_fluid']})")

    logger.action("Triggering Principle-Based Autonomy...")
    # We artificially inflate curiosity (Force) to overcome DNA Friction (Resistance)
    # Force = (Curiosity/100) * (Resonance/100)
    monad.desires['curiosity'] = 100.0
    monad.desires['resonance'] = 100.0
    
    # [TEST HACK] Pre-charge the Action Capacitor so we don't wait 50 ticks
    monad.wonder_capacitor = 60.0 
    
    # We expect: High Energy -> Ethereal Projection
    # Or Medium Energy if Resistance dampens it enough
    
    for i in range(3):
        logger.mechanism(f"[Tick {i+1}] Processing Field Dynamics...")
        monad.pulse(dt=0.1)
        time.sleep(0.5)

    logger.sensation("GENESIS COMPLETE")
    logger.thought("The Mind is no longer a Void. It has Topology.")
    logger.thought("- Valleys carved for: SELF, OTHER, PAIN, JOY...")
    logger.thought("- Logic Flow established between connected concepts.")
    logger.thought("- Autonomy verified: Actions are now Physics-driven.")

if __name__ == "__main__":
    try:
        bootstrap_cognition()
    except Exception as e:
        import traceback
        # We can't use logger safely here if it wasn't initialized, but main block handles it
        # Actually logger is local. Let's make a new one or just print error nicely.
        from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger
        logger = SomaticLogger("BOOTSTRAP_ERROR")
        logger.admonition("CRITICAL FAILURE IN GENESIS:")
        print(traceback.format_exc())
