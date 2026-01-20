
"""
RESTORE MEMORY
==============
Phase 28: The Great Memory Audit
Re-ingests fragmented CodeDNA into the Hypersphere.

"Gather the scattered light."
"""

import os
import json
import logging
from typing import Dict, Any

# Local Imports
from Core.Intelligence.Memory.hypersphere_memory import HypersphereMemory, HypersphericalCoord
from Core.System.Metabolism.zero_latency_portal import ZeroLatencyPortal
from Core.Foundation.Wave.wave_dna import WaveDNA

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RESTORE")

def restore_fragments():
    logger.info("🕵️ Starting Memory Restoration...")
    
    # 1. Initialize Memory
    memory = HypersphereMemory()
    logger.info(f"✅ Hypersphere Connected. Current Items: {memory._item_count}")
    
    # 2. Scan Raw Directory
    raw_dir = "c:/Elysia/data/Memory/Raw/Knowledge/CodeDNA"
    if not os.path.exists(raw_dir):
        logger.error(f"❌ Raw directory not found: {raw_dir}")
        return

    files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    logger.info(f"📂 Found {len(files)} memory fragments.")
    
    restored_count = 0
    
    # 3. Ingest Loop
    for filename in files:
        path = os.path.join(raw_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # data schema: {name, path, frequency, amplitude, phase, wavelength, ...}
            
            # Map Physics to Hypersphere Coords
            # Wavelength -> r (Depth) or Omega
            # Phase -> Theta/Phi
            
            wavelength = data.get('wavelength', 5.0)
            freq = data.get('frequency', 1.0)
            phase_val = 0.0
            
            # Simple heuristic mapping
            theta = (freq % 100) / 100.0 * 6.28  # 0~2pi
            phi = (wavelength % 10) / 10.0 * 6.28
            r = min(1.0, data.get('amplitude', 1.0))

            # Construct WaveDNA
            # The saved JSON likely doesn't have 7D fields, so we approximate or use defaults.
            # If data *is* a serialized WaveDNA, it might have them. 
            # But based on the file we saw earlier ("amplitude", "phase"), it's not a full 7D dump.
            # We create a generic DNA.
            dna = WaveDNA(
                label=data.get("name", "Unknown"),
                frequency=float(freq),
                physical=r,
                mental=wavelength / 10.0,
                causal=0.5
            )
            # The Rotor will call normalize(), which WaveDNA has.
            
            coord = HypersphericalCoord(theta=theta, phi=phi, psi=0.0, r=r)
            
            # Store in Hypersphere
            memory.store(
                data=f"CodeDNA: {data.get('name')}",
                position=coord,
                pattern_meta={
                    "dna": dna, # Pass Object, NOT dict
                    "omega": (freq, 0, 0),
                    "trajectory": "static",
                    "origin": "restored_fragment"
                }
            )
            restored_count += 1
            
            if restored_count % 100 == 0:
                print(f"   -> Restored {restored_count}/{len(files)}...")

        except Exception as e:
            # logger.warning(f"⚠️ Failed to restore {filename}: {e}")
            print(f"FAILED {filename}: {e}")
            
    # 4. Save
    logger.info("💾 Persisting Restored Memory...")
    memory.save_state()
    logger.info(f"🎉 Restoration Complete. Total Items in Hypersphere: {memory._item_count}")

if __name__ == "__main__":
    restore_fragments()
