"""
Core/Engine/Genesis/silicon_scholar_laws.py
===========================================
First Principles of Computing.

Simulating the fundamental logic of OS:
1. Scheduling (Time Management)
2. Paging (Space Management)
"""

import time
from typing import List, Dict, Any

def law_round_robin_scheduling(context, dt, intensity):
    """
    Principle: Fairness via Time Slicing.
    
    If multiple processes want the CPU, give each a small slice (Quantum).
    Context Requirements:
    - world: List of Monads (Processes)
    - global_state: 'cpu_owner', 'time_slice_remaining'
    """
    world = context["world"]
    # We need a scratchpad for the scheduler state. 
    # Since UniversalRotor doesn't persist state easily, we attach it to a 'System.Scheduler' monad effectively,
    # OR we use the context mutable dictionary if persistent.
    # The 'context' passed in `UniversalRotor.tick` is self.context from the rotor instance.
    
    # 1. Filter Processes (Ready Queue)
    ready_queue = [m for m in world if m.domain == "Process" and m.val > 0] # val = remaining burst time
    if not ready_queue:
        return

    # Initialize Scheduler State in Context if missing
    if "scheduler_state" not in context:
        context["scheduler_state"] = {
            "current_index": 0,
            "quantum": 2.0 * intensity, # Higher intensity = larger chunks? Or faster switching? Let's say Quantum size.
            "remaining_slice": 2.0 * intensity
        }
    
    state = context["scheduler_state"]
    
    # 2. Select Current Process
    if state["current_index"] >= len(ready_queue):
        state["current_index"] = 0
        
    current_proc = ready_queue[state["current_index"]]
    
    # 3. Execute (Consume Burst)
    # CPU work reduces burst time
    work_done = min(current_proc.val, dt * 10.0) # 10 units of work per tick
    current_proc.val -= work_done
    state["remaining_slice"] -= work_done
    
    # Visual Log
    current_proc.props["status"] = "RUNNING"
    for p in ready_queue:
        if p != current_proc: p.props["status"] = "WAITING"
        
    print(f"   ⚙️ [CPU] Running {current_proc.name} (Rem: {current_proc.val:.1f}) | Slice Rem: {state['remaining_slice']:.1f}")
    
    # 4. Context Switch Check
    if current_proc.val <= 0:
        print(f"   ✅ [CPU] Process {current_proc.name} COMPLETED.")
        current_proc.props["status"] = "DONE"
        # Remove from ready queue logic implicit in next tick's filter
        state["remaining_slice"] = 0 # Force switch
        
    if state["remaining_slice"] <= 0:
        # Time Questum Expired -> Switch
        state["current_index"] = (state["current_index"] + 1) % len(ready_queue)
        state["remaining_slice"] = state["quantum"]
        print("   🔄 [Sched] Context Switch! Next Process.")


def law_lru_eviction(context, dt, intensity):
    """
    Principle: Locality of Reference.
    
    If RAM is full, evict the Least Recently Used page.
    """
    world = context["world"]
    
    # Config
    MAX_RAM_PAGES = 3
    
    # 1. Access Pattern Simulation (Randomly touch a page)
    # In a real sim, a Process would touch a Page. Here we simulate 'Access' randomly or seq.
    # Let's say the 'Active Process' touches its pages.
    
    import random
    pages = [m for m in world if m.domain == "Memory"]
    if not pages: return
    
    # Simulate Access
    accessed_page = random.choice(pages)
    accessed_page.props["last_access"] = time.time()
    accessed_page.val += 1 # Access count
    
    # 2. Check Capacity
    active_pages = [p for p in pages if p.props.get("in_ram", False)]
    
    if len(active_pages) > MAX_RAM_PAGES:
        # 3. Eviction Logic (LRU)
        # Find page with oldest 'last_access'
        victim = min(active_pages, key=lambda p: p.props.get("last_access", 0))
        
        victim.props["in_ram"] = False
        print(f"   🗑️ [Mem] RAM Full! Evicting {victim.name} (LRU Logic)")
        
    # Ensure current accessed is in RAM
    if not accessed_page.props.get("in_ram", False):
        print(f"   📥 [Mem] Page Fault! Loading {accessed_page.name} into RAM.")
        accessed_page.props["in_ram"] = True

# ==============================================================================
# PHASE 17.5: THE ABYSS (Chaos & Concurrency)
# ==============================================================================

def law_dining_philosophers(context, dt, intensity):
    """
    Principle: Deadlock & Resource Contention.
    
    5 Philosophers (Threads) need 2 Forks (Resources) to eat.
    If everyone picks up the LEFT fork and waits for the RIGHT, they starve (Deadlock).
    """
    world = context["world"]
    
    # Identify Philosophers and Forks
    philosophers = [m for m in world if m.domain == "Thread"]
    forks = {m.name: m for m in world if m.domain == "Resource"}
    
    # Naive Logic: "I want to eat. I grab Left. I wait for Right."
    for p in philosophers:
        state = p.props.get("state", "THINKING")
        left_fork_name = p.props["left_fork"]
        right_fork_name = p.props["right_fork"]
        
        left_fork = forks[left_fork_name]
        right_fork = forks[right_fork_name]
        
        if state == "THINKING":
            # Decide to eat randomly
            import random
            if random.random() < 0.3 * intensity:
                p.props["state"] = "HUNGRY"
                print(f"   🤔 {p.name} is now HUNGRY.")
                
        elif state == "HUNGRY":
            # Try to grab Left Fork
            if left_fork.val == 0: # 0 = Free, 1 = Taken
                left_fork.val = 1
                p.props["state"] = "HOLDING_LEFT"
                print(f"   ⚡ {p.name} picked up {left_fork.name} (Left). Waiting for Right...")
            else:
                # Blocked
                pass
                
        elif state == "HOLDING_LEFT":
            # Try to grab Right Fork
            if right_fork.val == 0:
                right_fork.val = 1
                p.props["state"] = "EATING"
                p.props["eat_time"] = 2.0 # Eat for 2 ticks
                print(f"   🍝 {p.name} picked up {right_fork.name} (Right) and is EATING.")
            else:
                # DEADLOCK MOMENT: Holding one, waiting for other. 
                # If everyone is here, system freezes.
                pass
                
        elif state == "EATING":
            p.props["eat_time"] -= dt
            if p.props["eat_time"] <= 0:
                # Done eating. Release both.
                left_fork.val = 0
                right_fork.val = 0
                p.props["state"] = "THINKING"
                print(f"   😌 {p.name} finished eating and put down forks.")

def law_hardware_interrupts(context, dt, intensity):
    """
    Principle: Asynchrony & Kernel Traps.
    
    The CPU is executing User Code. Suddenly, data arrives on the Bus.
    The CPU MUST stop and handle the Interrupt.
    """
    world = context["world"]
    
    # Check for Hardware Signals (Monads with domain='Hardware')
    signals = [m for m in world if m.domain == "Hardware" and m.val > 0]
    
    if signals:
        # INTERRUPT!
        interrupt = signals[0]
        
        # 1. Save Context (Simulated)
        print(f"   🚨 [INTERRUPT] {interrupt.name} raised! Stopping User Mode...")
        
        # 2. Kernel Mode Execution (ISR)
        print(f"   🛡️ [Kernel] Handling {interrupt.name}...")
        interrupt.val = 0 # Handled (Clear signal)
        
        # 3. Restore Context
        print(f"   ▶️ [Resume] Returning to User Mode.")
    else:
        # Standard Execution
        pass
