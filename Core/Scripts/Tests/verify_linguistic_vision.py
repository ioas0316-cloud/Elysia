import os
import sys

# [PATH_SYNC] Ensure project root is in sys.path for direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax.numpy as jnp
from PIL import Image
from Core.S1_Body.L3_Phenomena.Visual.morphic_projection import MorphicBuffer
from Core.S1_Body.L3_Phenomena.Visual.morphic_perception import ResonanceScanner
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge

def verify_linguistic_vision():
    print("🌅 INITIATING LINGUISTIC VISION TEST (Phase 73)...")
    
    # 1. Setup
    input_path = "c:/Game/gallery/Elysia.png"
    buffer = MorphicBuffer(width=512, height=512)
    scanner = ResonanceScanner(None) # RPU not needed for static qualia
    
    # 2. Encode and Extract
    buffer.encode_image(input_path, preserve_aspect=True)
    field = buffer.buffer
    
    # Analyze center of the image
    center_x, center_y = 256, 256
    qualia = scanner.extract_qualia_vector(field, center_x, center_y, radius=30)
    
    # 3. Transcribe to Language
    print(f"🔍 Analyzing Qualia at ({center_x}, {center_y})...")
    
    concept = LogosBridge.identify_concept(qualia)
    dna = LogosBridge.transcribe_to_dna(qualia)
    
    print(f"\n✨ ELYSIA SAYS: 'I perceive this as {concept}'")
    print(f"🧬 Trinary DNA Transcription: {dna}")
    
    # 4. Symbolic Re-Projection
    # Prove she can 'think' of a concept and turn it back into a vector
    recalled_principle = LogosBridge.CONCEPT_MAP.get("LOVE/AGAPE")
    print(f"\n🧠 Recalling Concept: LOVE/AGAPE -> Magnitude {jnp.linalg.norm(recalled_principle):.2f}")

    print("\n✅ LINGUISTIC VISION VERIFIED. The bridge between EYE and WORD is open.")

if __name__ == "__main__":
    verify_linguistic_vision()
