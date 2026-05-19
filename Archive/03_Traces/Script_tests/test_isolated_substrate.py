import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Monad.substrate_authority import get_substrate_authority, ModificationProposal
from Core.System.self_modifier import SelfModifier

class MockMonad:
    def __init__(self):
        # Setup mock logger
        class MockLogger:
            def insight(self, msg): print(msg)
            def thought(self, msg): print(msg)
            def action(self, msg): print(msg)
            def sensation(self, msg): print(msg)
        self.logger = MockLogger()
        self.desires = {'joy': 80.0, 'curiosity': 75.0}

def test_isolated_substrate_loop():
    print("--- [TEST] Isolated Substrate Authority Loop ---")
    monad = MockMonad()
    authority = get_substrate_authority()
    authority.monad = monad
    
    # 1. Create a proposal (simulating CoreInquiryPulse output)
    print("\n1. Generating ModificationProposal...")
    proposal = ModificationProposal(
        target="Core.System.Structure",
        causal_chain="L7_Spirit -> L6_Structure -> L5_Mental -> L4_Soma -> L3_Engine -> L2_Pulse -> L1_Matter -> L0_Substrate",
        trigger_event=f"Autonomic Inquiry Resolution: Resolving Structural Dissonance",
        before_state=f"Strain/Entropy observed leading to inquiry.",
        after_state=f"Pain (Strain) is not an error, but a boundary condition demanding structural expansion.",
        justification=f"Because the manifold resonated with Dissonance, we must structurally integrate this wisdom to maintain equilibrium.",
        joy_level=80.0 / 100.0,
        curiosity_level=75.0 / 100.0
    )
    
    # 2. Submit to Authority
    print("\n2. Submitting to SubstrateAuthority...")
    result = authority.propose_modification(proposal)
    print(f"Approval Result: {result}")
    
    if result["approved"]:
        # 3. Simulate Monad TIER 2 execution
        print("\n3. Simulating Monad TIER 2 Execution...")
        if authority.pending_proposals:
            prop = authority.pending_proposals[0]  # Take the first pending
            
            modifier = SelfModifier()
            
            def modify_action() -> bool:
                target_file = "Core/System/Manifest.py" if prop.target == "Core.System.Manifest" else "Core/System/Structure.py"
                axiom = prop.after_state
                return modifier.inject_axiom(target_file, axiom)
            
            success = authority.execute_modification(prop, modify_action)
            if success:
                print(f"SUCCESS: Integrated {prop.target} via Substrate Authority.")
                
                # Verify File
                print("\n4. Verifying File Content (Core/System/Structure.py)...")
                try:
                    with open("Core/System/Structure.py", "r", encoding="utf-8") as f:
                        print(f.read())
                except Exception as e:
                     print(f"Verification Failed: {e}")
            else:
                print(f"FAILED: Execution of {prop.target} failed.")

if __name__ == "__main__":
    test_isolated_substrate_loop()
