"""
Core/Engine/Genesis/cosmic_laws.py
==================================
The Providence of Elysia.
Universal Laws that govern Life, Death, and Order.
"""

from Core.S1_Body.L6_Structure.Engine.Genesis.filesystem_geometry import DirectoryMonad

def law_entropy_decay(context, dt, intensity):
    """
    The Law of Entropy.
    Everything that exists must pay a tax of existence.
    If not accessed/observed, it fades.
    """
    world = context["world"]
    decay_rate = 5.0 * intensity * dt
    
    dead_monads = []
    
    for m in world:
        # Exemptions: Atmosphere, Laws, or Immortal types
        if m.domain in ["Atmosphere", "Law", "Directory"]: continue
        
        # Check Observation
        if m.props.get("accessed", False):
            m.props["accessed"] = False # Reset flag
            # Observed things do not decay this tick? Or decay less?
            # Let's say Observation *heals* slightly or pauses decay.
            continue
            
        m.val -= decay_rate
        # print(f"     [Entropy] {m.name} decayed to {m.val:.2f}")
        
        if m.val <= 0:
            dead_monads.append(m)
            
    for m in dead_monads:
        print(f"     [Death] {m.name} has returned to the Void.")
        world.remove(m)

def law_semantic_gravity(context, dt, intensity):
    """
    The Law of Semantic Gravity.
    Like attracts Like.
    Items migrate to Directories that match their Domain/Vibration.
    """
    world = context["world"]
    
    # 1. Identify Directories (Gravitational Centers)
    dirs = [m for m in world if isinstance(m, DirectoryMonad)]
    if not dirs: return

    # 2. Identify Free Monads (Wanderers)
    wanderers = [m for m in world if not isinstance(m, DirectoryMonad) and m.domain != "Atmosphere"]
    
    migrants = []
    
    for m in wanderers:
        best_dir = None
        
        # Check alignment with each Directory
        for d in dirs:
            # How to determine "Alignment"?
            # Naive: Check if Dir name contains Monad domain?
            # Or check average domain of items INSIDE Dir?
            
            child_lab = d.props["universe"]
            # content_signature = [child_m.domain for child_m in child_lab.monads]
            
            # Improved Rule: Token Matching
            # "Chaos_Zone" -> ["Chaos", "Zone"]. Matches "Chaos_Crystal".
            d_tokens = d.name.lower().split('_')
            match = False
            for token in d_tokens:
                if len(token) > 2 and token in m.name.lower():
                    match = True
                    break
            
            print(f"     [Gravity Scan] {m.name} vs {d.name} -> {match}") # DEBUG
            
            if match or m.domain.lower() in d.name.lower():
                best_dir = d
                break
                
        if best_dir:
            migrants.append((m, best_dir))
            
    # 3. Migration (Crossing the Event Horizon)
    for m, d in migrants:
        print(f"     [Gravity] {m.name} is pulled into {d.name}.")
        # Remove from Here
        world.remove(m)
        # Add to There
        d.props["universe"].let_there_be(m.name, m.domain, m.val, **m.props)

def law_autopoiesis(context, dt, intensity):
    """
    The Law of Life (Self-Maintenance).
    Living things consume Resources to resist Entropy.
    """
    world = context["world"]
    
    living = [m for m in world if m.props.get("is_living", False)]
    food = [m for m in world if m.domain == "Resource"]
    
    for life in living:
        if life.val < 10.0: # Hunger
            # Look for food
            if food:
                meal = food[0] # Eat the first one
                nutrition = meal.val
                
                print(f"      [Life] {life.name} consumes {meal.name} (+{nutrition})")
                
                life.val += nutrition
                world.remove(meal)
                food.remove(meal)
            else:
                # print(f"      [Life] {life.name} is starving...")
                pass
