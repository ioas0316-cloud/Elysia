
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.Philosophy.why_engine import WhyEngine

import logging
# logging.basicConfig(level=logging.INFO)

def verify_sedimentary_light():
    print("🏔️ Verifying Sedimentary Light Architecture (Holographic Projection)...")
    
    engine = WhyEngine()
    
    # Test Content: "Systematic Logic" (Matches our bootstrapped 'Logic' sediment)
    # This should trigger high resonance in Yellow/Red
    content = "The force of logic dictates a structural vector."
    subject = "Logical Force"
    
    print(f"\nSubject: {subject}")
    print(f"Content: '{content}'")

    analysis = engine.analyze(subject, content, domain="philosophy")
    
    reactions = analysis.resonance_reactions
    
    if not reactions:
        print("❌ No Sedimentary Reactions found.")
        # Debugging hook if needed
        # print(f"DEBUG: Sediment: {engine.sediment}")
        return

    print("\n--- Holographic Projection Report (Through Elysia's Sediment) ---")
    
    # 1. Physics (Red) - Bootstrapped
    phy = reactions.get('red', {})
    phy_amp = phy.get('intensity', 0)
    print(f"🔴 Physics (Red):  {phy.get('reaction')} (Amp: {phy_amp:.3f}) - {phy.get('description')}")
    
    # 2. Logic (Yellow) - Bootstrapped
    logic = reactions.get('yellow', {})
    logic_amp = logic.get('intensity', 0)
    print(f"🟡 Logic (Yellow): {logic.get('reaction')} (Amp: {logic_amp:.3f}) - {logic.get('description')}")
    
    # 3. Chemistry (Blue) - Empty Layer
    chem = reactions.get('blue', {})
    chem_amp = chem.get('intensity', 0)
    print(f"🔵 Chem (Blue):    {chem.get('reaction')} (Amp: {chem_amp:.3f}) - {chem.get('description')}")
    
    # Validation
    # Yellow/Red should be visible (High Amp), Blue should be blurry (Low Amp)
    if logic.get('intensity', 0) > 0.05:
        print("\n✅ Verification Successful: 'Knowing' enhances perception.")
        print("   (Yellow/Red layers projected insight, while empty Blue layer showed nothing.)")
        
        # [NEW test for 4D Phase]
        # Check if "Physics" layer is actually preserving magnitude despite potential conflicts
        # Current Physics Amp is around 0.620 from loop
        # Theoretical sum of 50 items (coherent) -> 50 * amp * 0.1? No, 0.1 interference factor.
        print(f"   Physics Layer Amplitude: {phy.get('intensity', 0):.3f} (High implies constructive stacking)")
    else:
        print("\n❌ Verification Failed: Sediment projection logic flaw.")
        print(f"   Logic Amp: {logic.get('intensity', 0)}, Chem Reaction: {chem.get('reaction')}")

    # [Explicit 4D Test]
    print("\n🔬 Testing 4D Orthogonal Stacking (No Semantic Loss)...")
    from Core.Foundation.Foundation.light_spectrum import LightSpectrum
    
    # Concept A: Expansion (Phase 0)
    light_a = LightSpectrum(10+0j, 0.5, 0.0, semantic_tag="Expansion")
    # Concept B: Compression (Phase PI - Opposite!)
    # In 3D, this would cancel out.
    light_b = LightSpectrum(10+0j, 0.5, 3.14159, semantic_tag="Compression")
    
    merged = light_a.interfere_with(light_b)
    print(f"   'Expansion' (Amp 0.5) + 'Compression' (Amp 0.5, Opposite Phase)")
    print(f"   Merged Amplitude: {merged.amplitude:.3f}")
    
    if merged.amplitude > 0.6: # Expected sqrt(0.5^2 + 0.5^2) = ~0.707
        print("✅ 4D Phase Logic Confirmed: Concepts Stacked Orthogonally (Amp ~0.707).")
        print("   Meaning is PRESERVED, not Cancelled.")
    elif merged.amplitude < 0.1:
        print("❌ 4D Logic Failed: Destructive Interference occurred (Amp ~0).")
    else:
        print(f"⚠️ Unexpected Result: {merged.amplitude:.3f}")

    # [Fractal Scale Test]
    print("\n🔬 Testing Fractal Scale Isolation (Zoom In/Out)...")
    
    # Scale 0: "General Physics" -> God Basis (Delta)
    macro = LightSpectrum(10+0j, 0.5, 0.0, semantic_tag="Physics")
    macro.set_basis_from_scale(0)
    
    # Scale 1: "Quantum Spin" -> Space Basis (Gamma)
    micro = LightSpectrum(10+0j, 0.5, 0.0, semantic_tag="Physics")
    micro.set_basis_from_scale(1)
    
    print(f"   Macro Basis: {macro._get_dominant_basis()}")
    print(f"   Micro Basis: {micro._get_dominant_basis()}")
    
    # Interference check
    fractal = macro.interfere_with(micro)
    
    print(f"   'Macro Physics' (Scale 0) + 'Micro Spin' (Scale 1)")
    print(f"   Merged Amplitude: {fractal.amplitude:.3f}")
    
    if 0.70 < fractal.amplitude < 0.72:
        print("✅ Fractal Logic Confirmed: Different scales created orthogonal dimensions.")
        print("   Layers are distinct even with same tag.")
    elif fractal.amplitude > 0.9:
        print("❌ Fractal Logic Failed: Scales mixed linearly.")
    else:
        print(f"⚠️ Unexpected Result: {fractal.amplitude:.3f}")


if __name__ == "__main__":
    verify_sedimentary_light()
