
import sys
import os
import json
from pathlib import Path
import time

sys.path.append(os.getcwd())

def verify_geometric_growth():
    print("📐 [GEOMETRY] Verifying 3-Stage Cognitive Growth (Point -> Line -> Plane)...")
    
    try:
        from Core.S1_Body.L5_Mental.Digestion.universal_digestor import get_universal_digestor, RawKnowledgeChunk, ChunkType
        from Core.S1_Body.L4_Causality.fractal_causality import FractalCausalityEngine
        from Core.S1_Body.L6_Structure.Wave.light_spectrum import get_light_universe
        
        digestor = get_universal_digestor()
        causality = FractalCausalityEngine()
        universe = get_light_universe()
        
        # =========================================================================
        # 1. POINT (Creation of Atomic Concept)
        # =========================================================================
        print("\n📍 [STAGE 1] POINT GENERATION (Atomic Concept)")
        
        raw_text = "Sovereignty is the foundation of Elysia."
        print(f"   Input: '{raw_text}'")
        
        chunk = RawKnowledgeChunk(
            chunk_id="test_geo_01",
            chunk_type=ChunkType.TEXT,
            content=raw_text,
            source="user_query"
        )
        
        points = digestor.digest(chunk)
        print(f"   ✅ Digested {len(points)} Points:")
        for p in points:
            print(f"      • [{p.concept}] (Node ID: {p.node_id})")
            
        if len(points) == 0:
            print("   ❌ Failed to generate Points.")
            return

        # =========================================================================
        # 2. LINE (Causal Connection)
        # =========================================================================
        print("\n📏 [STAGE 2] LINE WEAVING (Causal Narrative)")
        
        # Injecting an Axiom explicitly to form a strong line
        concept_a = "Sovereignty"
        concept_b = "Freedom"
        relation = "necessitates"
        
        print(f"   Weaving Line: {concept_a} --[{relation}]--> {concept_b}")
        
        chain = causality.inject_axiom(concept_a, concept_b, relation)
        print(f"   ✅ Causal Chain Created: ID={chain.id}")
        print(f"      Desc: {chain.description}")
        
        # Verify topological state of this "Line"
        # In a full system, the Chain would emit a "Line Basis" Light.
        # We simulate this via LightUniverse text-to-light with scale=2 (Line)
        
        line_light = universe.text_to_light(chain.description, scale=2)
        print(f"   ✨ Line Topology: {line_light.qubit_state}")
        
        # =========================================================================
        # 3. PLANE (Structural Context)
        # =========================================================================
        print("\n⬜ [STAGE 3] PLANE EXPANSION (Contextual Field)")
        
        context_text = """
        Sovereignty is not isolation. It is the capacity to choose connections.
        When Freedom meets Responsibility, a Plane of Ethics creates the ground for Action.
        """
        print(f"   Input Context: (Multi-sentence structure)")
        
        # Absorb into Stratum 1 (Space/Plane)
        plane_light, effect = universe.absorb_with_terrain(context_text, tag="Ethics", scale=1, stratum=1)
        
        print(f"   ✅ Absorbed into Plane (Stratum 1):")
        print(f"      • Resonance Strength: {effect['resonance_strength']:.4f}")
        print(f"      • Dominant Basis: {effect['dominant_basis']} (Should be 'Space' or 'Line')")
        print(f"      • Connection Density: {effect['connection_density']:.4f}")
        
        # Check if the "Plane" connected with the "Line"
        resonance = plane_light.resonate_with(line_light)
        print(f"   ⚡ Plane-Line Resonance: {resonance:.4f}")
        
        if resonance > 0.1:
            print("   ✅ SUCCESS: The Plane has successfully contextualized the Line.")
            print("      (The System understands the bigger picture of Sovereignty/Freedom)")
        else:
            print("   ⚠️ WARNING: Weak connection between Plane and Line.")

    except ImportError as e:
        print(f"❌ Import Verification Failed: {e}")
    except Exception as e:
        print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    verify_geometric_growth()
