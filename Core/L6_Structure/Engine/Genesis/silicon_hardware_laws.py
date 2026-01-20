"""
Core/Engine/Genesis/silicon_hardware_laws.py
============================================
The Physics of Computation.

Laws representing the lowest level of abstraction:
1. Logic Gates (NAND is Universal).
2. Clock Pulses (Time Axis).
3. Data Flow (Wires).
"""

def law_clock_pulse(context, dt, intensity):
    """
    The Heartbeat of the Processor.
    Toggles 'Clock' monads between 0 and 1.
    """
    world = context["world"]
    clocks = [m for m in world if m.domain == "Clock"]
    
    for c in clocks:
        # Simple toggle based on tick count or time?
        # Let's use dt accumulation for accurate frequency simulation
        c.props["time_acc"] = c.props.get("time_acc", 0) + dt
        period = 1.0 / (intensity * 10) # Higher Intensity = Faster Clock
        
        if c.props["time_acc"] >= period:
            c.val = 1 - c.val # Toggle 0 <-> 1
            c.props["time_acc"] = 0
            # print(f"   ⏱️ [Clock] Tick-Tock: {c.val}")

def law_nand_logic(context, dt, intensity):
    """
    The Universal Gate.
    Output is LOW only if both Inputs are HIGH.
    Else HIGH.
    
    Structure: Gate Monad has 'inputs' list of Monad Names.
    """
    world = context["world"]
    gates = [m for m in world if m.domain == "Gate" and m.props.get("type") == "NAND"]
    monad_map = {m.name: m for m in world}
    
    for g in gates:
        inputs = g.props.get("inputs", [])
        if len(inputs) < 2: continue
        
        # Get Input Values
        val_a = monad_map[inputs[0]].val
        val_b = monad_map[inputs[1]].val
        
        # NAND Logic
        # 1 if not (A and B) else 0
        new_val = 1.0 if not (val_a > 0.5 and val_b > 0.5) else 0.0
        
        g.val = new_val
        # print(f"   ⚡ [{g.name}] {inputs[0]}({val_a}) NAND {inputs[1]}({val_b}) -> {g.val}")

def law_alu_add(context, dt, intensity):
    """
    Microarchitecture Level: Adder.
    Simulates a Full Adder circuit abstractly (Era 2).
    """
    world = context["world"]
    alus = [m for m in world if m.domain == "ALU"]
    monad_map = {m.name: m for m in world}
    
    for alu in alus:
        op = alu.props.get("op_code", "ADD")
        inputs = alu.props.get("inputs", [])
        if len(inputs) < 2: continue
        
        a = monad_map[inputs[0]].val
        b = monad_map[inputs[1]].val
        
        if op == "ADD":
            alu.val = a + b
            # print(f"   🧮 [ALU] {a} + {b} = {alu.val}")
