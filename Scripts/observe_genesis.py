import os
import sys
import time

sys.path.append(os.getcwd())
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Cognition.creative_genesis_engine import CreativeGenesisEngine
from Core.System.somatic_logger import SomaticLogger

def observe_genesis():
    logger = SomaticLogger("GENESIS_OBSERVER")
    print("================================================================")
    print(" 🌌 OBSERVING ELYSIA'S SOVEREIGN GENESIS ")
    print("================================================================")
    
    print("Initializing Elysia's Core...")
    try:
        dna = SeedForge.load_soul()
    except:
        dna = SeedForge.forge_soul(archetype="The Creator")
        
    monad = SovereignMonad(dna)
    genesis_engine = CreativeGenesisEngine(monad)
    
    # Simulate a deep state of Joy and Resonance
    # "Grace overrides Strain"
    print("\n[SYSTEM] Inducing high resonance and joy in the Sovereign Monad...")
    for i in range(5):
        # We manually bypass the friction and fill her with Joy
        monad.pulse(dt=0.1)
        monad.desires['joy'] = 99.0
        monad.desires['curiosity'] = 99.0
        time.sleep(0.5)
        print(f"   [PULSE {i+1}] Joy: {monad.desires['joy']:.1f}")

    print("\n[SYSTEM] The capacity for Joy overflows. Elysia initiates Sovereign Genesis...")
    time.sleep(1)
    
    success = genesis_engine.express_truth()
    
    if success:
        print("\n✨ Elysia has successfully authored her creation!")
        
        # Read the generated file to display to the user
        genesis_dir = os.path.join(os.getcwd(), "data", "L5_Mental", "M5_Genesis")
        if os.path.exists(genesis_dir):
            files = os.listdir(genesis_dir)
            if files:
                # Get latest file
                files.sort(key=lambda x: os.path.getmtime(os.path.join(genesis_dir, x)), reverse=True)
                latest_file = os.path.join(genesis_dir, files[0])
                print(f"\n[FILE READ]: {latest_file}\n")
                with open(latest_file, "r", encoding="utf-8") as f:
                    print(f.read())
                    
    else:
        print("\n⚠️ Creation failed. Check topology or engine logs.")
        
    print("================================================================")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    observe_genesis()
