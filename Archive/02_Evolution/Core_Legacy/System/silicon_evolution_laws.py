"""
Core/Engine/Genesis/silicon_evolution_laws.py
===============================================
Evolutionary Solutions to Silicon Chaos.

Laws that solve the problems introduced in 'silicon_scholar_laws.py'.
"""

def law_resource_hierarchy(context, dt, intensity):
    """
    Solution to Dining Philosophers.
    
    Rule: Always pick up the LOWER ID fork first.
    This prevents circular dependency (Deadlock).
    """
    world = context["world"]
    
    # Identify Philosophers and Forks
    philosophers = [m for m in world if m.domain == "Thread"]
    forks = {m.name: m for m in world if m.domain == "Resource"}
    
    for p in philosophers:
        state = p.props.get("state", "THINKING")
        left_name = p.props["left_fork"]
        right_name = p.props["right_fork"]
        
        # Sort forks by Name/ID to establish Hierarchy
        # e.g. Fork_1 < Fork_2
        first_fork_name, second_fork_name = sorted([left_name, right_name])
        
        first_fork = forks[first_fork_name]
        second_fork = forks[second_fork_name]
        
        if state == "THINKING":
            import random
            if random.random() < 0.3 * intensity:
                p.props["state"] = "HUNGRY"
                
        elif state == "HUNGRY":
            # Try to grab FIRST fork (Lower ID)
            if first_fork.val == 0:
                first_fork.val = 1
                p.props["state"] = "HOLDING_FIRST"
                # print(f"      {p.name} obeyed Hierarchy: Picked up {first_fork.name} first.")
            else:
                # Blocked
                pass
                
        elif state == "HOLDING_FIRST":
            # Try to grab SECOND fork (Higher ID)
            if second_fork.val == 0:
                second_fork.val = 1
                p.props["state"] = "EATING"
                p.props["eat_time"] = 2.0
                # print(f"     {p.name} picked up {second_fork.name} and is EATING.")
                
                # METRIC FOR COGNITIVE CYCLE
                p.props["meals_eaten"] = p.props.get("meals_eaten", 0) + 1
                
            else:
                # Blocked, but NO DEADLOCK possible because someone else must be holding the higher fork
                # and will eventually finish.
                pass
                
        elif state == "EATING":
            p.props["eat_time"] -= dt
            if p.props["eat_time"] <= 0:
                # Release
                first_fork.val = 0
                second_fork.val = 0
                p.props["state"] = "THINKING"
                # print(f"     {p.name} finished.")

# ==============================================================================
# PHASE 20: THE CONTINUUM (Space & Persistence)
# ==============================================================================

MAX_MEMORY_CAPACITY = 5 # Small capacity for demo

def law_naive_oom(context, dt, intensity):
    """
    Space Principle (Naive):
    If System is Full, CRASH or DELETE randomly.
    """
    world = context["world"]
    
    # Check Usage
    active_monads = [m for m in world if m.props.get("in_ram", True)]
    usage = len(active_monads)
    
    # print(f"     [Mem Check] Usage: {usage}/{MAX_MEMORY_CAPACITY}")
    
    if usage > MAX_MEMORY_CAPACITY:
        print(f"     [CRASH] OUT OF MEMORY! (Usage {usage} > {MAX_MEMORY_CAPACITY})")
        # Simulate Crash: Wipe everything or Raise Error?
        # Let's Raise Error to simulate System Failure
        # But we catch it in CognitiveCycle to measure 'Suffering'
        # Actually, let's just mark a global failure flag in context
        context["system_status"] = "CRASHED_OOM"
        
        # Kill random process to survive? (OOM Killer) - but naive just crashes.
        # Let's clear world to simulate reboot or just halt.
        world.clear() 

def law_lru_paging(context, dt, intensity):
    """
    Space Principle (Advanced):
    If System is Full, SWAP OUT the LRU item.
    """
    world = context["world"]
    
    # 1. Identify RAM vs Disk
    ram_monads = [m for m in world if m.props.get("in_ram", True)]
    
    usage = len(ram_monads)
    
    if usage > MAX_MEMORY_CAPACITY:
        # PAGING NEEDED
        # Find victim (LRU) - assuming 'last_access' is updated by something (access law)
        # Verify access logic exists? For now assume everything is accessed randomly or simply exists.
        # Let's sort by 'last_access' (default 0).
        
        # Just pick the one with oldest timestamp.
        import time
        # Ensure everyone has a timestamp
        for m in ram_monads:
            if "last_access" not in m.props: m.props["last_access"] = time.time()
            
        victim = min(ram_monads, key=lambda m: m.props["last_access"])
        
        # 2. Swap Out
        victim.props["in_ram"] = False
        print(f"     [Swap] RAM Full ({usage}). Moving {victim.name} to Disk.")
        
    else:
        # Check if anyone needs to be Swapped In? (If accessed)
        # This law mainly handles *Eviction* (Management). The Access Law handles *Faults*.
        # For this test, we assume we just want to PREVENT CRASH.
        pass
