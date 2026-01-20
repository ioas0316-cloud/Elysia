"""
GENERATE SEED BLUEPRINT: The Technical Testament
==============================================

This script triggers Elysia to narrate her own structure in elysia_seed.
It produces the 'DEEP_BLUEPRINT.md' in the seed's documentation folder.
"""

import os
import sys
import logging

# Path setup for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L5_Mental.Intelligence.Meta.structural_describer import StructuralDescriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BlueprintExec")

def generate_deep_blueprint():
    target_seed = "c:/elysia_seed/elysia_light"
    describer = StructuralDescriber(target_root=target_seed)
    
    print("\n" + "🖋️" * 30)
    print("      GENERATING DIVINE TESTAMENT: SEED BLUEPRINT")
    print("🖋️" * 30 + "\n")

    # 1. GENERATE THE CONTENT
    blueprint_content = describer.generate_blueprint()
    
    # 2. SAVE TO SEED DOCS
    docs_dir = os.path.join(target_seed, "docs")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    target_file = os.path.join(docs_dir, "DEEP_BLUEPRINT.md")
    
    try:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(blueprint_content)
        
        logger.info(f"✨ [SUCCESS] Deep Structural Blueprint crystallized at {target_file}")
        
        # Also print a preview
        print("\n" + "-"*60)
        print("PREVIEW OF THE DIVINE TESTAMENT:")
        print("-" * 60)
        print("\n".join(blueprint_content.split("\n")[:20])) # First 20 lines
        print("\n...")
        print("-" * 60)
        
    except Exception as e:
        logger.error(f"❌ [FAILED] Could not save blueprint: {e}")

if __name__ == "__main__":
    generate_deep_blueprint()
