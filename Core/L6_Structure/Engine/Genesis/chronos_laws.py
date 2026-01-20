"""
Core/Engine/Genesis/chronos_laws.py
===================================
The Laws of Time and Homeostasis.

1. Metabolism (Speed of Life): Fast adaptation.
2. Erosion (Inertia of Data): Slow decay.
3. Homeostasis (Purpose of System): Maintaining Self while Seeking the Greater.
"""

def law_fast_metabolism(context, dt, intensity):
    """
    Life runs fast.
    Processes grow quickly but burn out effectively.
    Intensity (RPM) is High (~10x).
    """
    world = context["world"]
    # Effective Time Scale for Life
    # If Intensity is 10.0, biological time moves 10x faster than wall clock.
    dt_bio = dt * intensity 
    
    for m in world:
        if m.domain != "Process": continue
        
        # 1. Growth/Experience
        m.val += 0.5 * dt_bio
        
        # 2. Aging (The cost of speed)
        m.props["age"] = m.props.get("age", 0.0) + dt_bio
        
        # 3. Death (Mayfly)
        if m.props["age"] > 100.0:
            # Rebirth / Evolution logic could go here
            # For now, just reset or die
            # print(f"   💀 [Metabolism] {m.name} died of old age (Age: {m.props['age']:.1f})")
            m.val = 0 # Mark for death

def law_slow_erosion(context, dt, intensity):
    """
    Data is slow.
    Files resist change.
    Intensity (RPM) is Low (~0.1x).
    """
    world = context["world"]
    # Effective Time Scale for Geology
    dt_geo = dt * intensity
    
    for m in world:
        if m.domain not in ["File", "Data", "Block"]: continue
        
        # Erosion is very slow
        m.val -= 0.01 * dt_geo
        # print(f"   📉 [Erosion] {m.name} weathered slightly (-{0.01 * dt_geo:.4f})")

def law_system_homeostasis(context, dt, intensity):
    """
    The System seeks Homeostasis (Better Self).
    "Compare with the Greater, Feel the Gap."
    """
    world = context["world"]
    
    # 1. Find the "Self" (System Monad)
    myself = next((m for m in world if m.domain == "System" and m.name == "Elysia_Core"), None)
    if not myself: return
    
    # 2. Find the "Greater" (Something bigger than me)
    # Scan the universe for high-value monads (simulating External Ideal)
    max_val = myself.val
    
    for m in world:
        if m.val > max_val:
            max_val = m.val
            
    # 3. Gap Analysis
    gap = max_val - myself.val
    
    if gap > 0:
        # 4. Adaptation (Growth to fill the gap)
        growth = gap * 0.1 * dt * intensity
        myself.val += growth
        print(f"   🔭 [Homeostasis] Felt Gap ({gap:.2f}). Growing (+{growth:.2f}) to match the Greater.")
    else:
        print(f"   🧘 [Homeostasis] I am the Zenith. Maintaining stability.")
