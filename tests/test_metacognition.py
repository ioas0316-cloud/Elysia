from Core.Foundation.planning_cortex import PlanningCortex, SophiaBlueprint

def test_metacognition():
    print("🧪 Testing Metacognition (Sophia's Mirror)...")
    
    # 1. Initialize Architect
    architect = PlanningCortex()
    
    # 2. Simulate Current State (Imperfect)
    current_state = {
        "imagination": False, # Missing
        "memory_depth": 2,    # Shallow
        "empathy": True,
        "quantum_thinking": True
    }
    
    print(f"   📊 Current State: {current_state}")
    
    # 3. Audit Capabilities
    # Note: We added a wrapper in PlanningCortex, so we can call it directly
    gaps = architect.audit_capabilities(current_state)
    
    if gaps:
        print(f"   ❌ Gaps Found: {len(gaps)}")
        for gap in gaps:
            print(f"      - {gap}")
    else:
        print("   ✅ No Gaps Found.")
        
    # 4. Generate Evolution Plan
    plan = architect.generate_evolution_plan(gaps)
    print(f"\n   🧬 Generated Plan:\n{plan}")
    
    # Verification
    if "Ignite HolographicCortex" in plan and "Deepen Hippocampal Index" in plan:
        print("✅ PASS: Evolution Plan correctly addresses gaps.")
    else:
        print("❌ FAIL: Plan missing key evolutionary tasks.")

if __name__ == "__main__":
    test_metacognition()
