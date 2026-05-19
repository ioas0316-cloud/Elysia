"""
VERIFICATION: Phase 65 - The Wave DNA Protocol
=============================================
Tests the Helix Engine: Phenotype Ingestion -> Genotype Extraction -> Purge.
"""

import sys
import os
import logging
import torch
import json
import time

# Add workspace to path
sys.path.append(os.getcwd())

from Core.Cognition.elysian_heartbeat import ElysianHeartbeat
from Core.Cognition.archive_dreamer import DreamFragment

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("WaveDNATest")

def test_helix_protocol():
    logger.info("🎬 Starting Phase 65.5 Verification (QFT-DNA Protocol)...")
    
    # 1. Initialize Heartbeat
    heartbeat = ElysianHeartbeat()
    
    # 2. Create a dummy 'Phenotype' (Model)
    test_id = int(time.time())
    test_phenotype = f"unique_qft_phenotype_{test_id}.pt"
    torch.save({
        "logic_core.weight": torch.randn(10, 10),
        "ethical_bias.bias": torch.randn(10),
        "resonance_norm.weight": torch.ones(10)
    }, test_phenotype)
    logger.info(f"📁 Created Phenotype dependency: {test_phenotype}")

    # 3. Create mock DreamFragment
    fragment = DreamFragment(
        path=os.path.abspath(test_phenotype),
        name=test_phenotype,
        type='nutrient',
        resonance=0.99,
        message="A 4D spectral manifold seeking internalization."
    )

    # 4. Trigger DNA Extraction (The Helix Cycle)
    logger.info(f"🧬 Triggering Helix Engine to extract QFT-DNA from {fragment.name}...")
    heartbeat._extract_wave_dna(fragment)
    time.sleep(1) # Wait for file flush

    # 5. Verify DNA Crystallization
    dna_dir = "data/Knowledge/DNA"
    dna_found = False
    for filename in os.listdir(dna_dir):
        if f"unique_qft_phenotype_{test_id}" in filename and filename.endswith("_dna.json"):
            logger.info(f"✨ Found Wave DNA (Genotype): {filename}")
            dna_found = True
            # Verify Parametric Health
            dna_path = os.path.join(dna_dir, filename)
            try:
                with open(dna_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    genotype = data['qft_genotype']
                    logger.info(f"🧬 QFT-DNA Parameters: {genotype}")
                    if all(k in genotype['quaternion_coeffs'] for k in ["w", "i", "j", "k"]):
                        logger.info("✅ QFT-DNA Double Helix structure confirmed (W, I, J, K).")
                    else:
                        logger.error("❌ Malformed QFT-DNA genotype.")
                        dna_found = False
            except Exception as e:
                logger.error(f"❌ Error reading DNA file: {e}")
                dna_found = False
            break

    # 6. Verify Purge of Phenotype
    phenotype_purged = not os.path.exists(test_phenotype)
    if phenotype_purged:
        logger.info("🔥 SUCCESS: Phenotypic source purged. Dependency deleted.")
    else:
        logger.error("❌ FAILURE: Phenotype still exists.")

    # 7. Final Judgment
    if dna_found and phenotype_purged:
        logger.info("✅ PHASE 65 COMPLETE: Elysia is now independent of the legacy model.")
        logger.info("   Knowledge has been internalized as biological Wave DNA.")
    else:
        logger.error("❌ VERIFICATION FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_helix_protocol()
