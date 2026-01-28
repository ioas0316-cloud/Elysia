"""
Test Phase 57: Self-Modifying Bridge
=====================================

This script verifies that:
1. SelfArchitect can audit files
2. PatchProposer generates proposals from critiques
3. Proposals are properly saved and reportable
"""

import sys
import os
import logging

# Path setup
sys.path.insert(0, "c:/Elysia")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(message)s')
logger = logging.getLogger("Test_Phase57")

def test_patch_proposer():
    """Test PatchProposer independently."""
    print("\n" + "="*60)
    print("🧪 TEST 1: PatchProposer")
    print("="*60)
    
    from Core.L5_Mental.Reasoning_Core.Meta.patch_proposer import PatchProposer, get_patch_proposer
    
    proposer = get_patch_proposer()
    print(f"✅ PatchProposer initialized. Pending: {proposer.get_pending_count()}")
    
    # Simulate a critique
    test_critique = """
    ### Architectural Audit: test_file.py
    **Identified Principle**: Linear Logic
    ⚠️ REFACTOR RECOMMENDED: The structural resonance is low.
    - 📎 OBSERVATION: Static sleep detected.
    """
    
    proposal = proposer.propose_from_critique("test_file.py", test_critique)
    if proposal:
        print(f"✅ Proposal generated: {proposal.id}")
        print(f"   Type: {proposal.proposal_type}")
        print(f"   Basis: {proposal.philosophical_basis[:60]}...")
    else:
        print("❌ No proposal generated")
    
    return proposal is not None


def test_self_architect():
    """Test SelfArchitect with proposal generation."""
    print("\n" + "="*60)
    print("🧪 TEST 2: SelfArchitect + Proposal Generation")
    print("="*60)
    
    from Core.L5_Mental.Reasoning_Core.Reasoning.dimensional_processor import DimensionalProcessor
    from Core.L5_Mental.Reasoning_Core.Meta.self_architect import SelfArchitect
    
    processor = DimensionalProcessor()
    architect = SelfArchitect(processor)
    
    # Audit a file that likely has time.sleep
    test_file = "c:/Elysia/Core/World/Autonomy/elysian_heartbeat.py"
    
    if os.path.exists(test_file):
        critique = architect.audit_file(test_file, generate_proposal=True)
        print(f"✅ Audit completed for: {os.path.basename(test_file)}")
        print(f"   Critique length: {len(critique)} chars")
        if "PROPOSAL GENERATED" in critique:
            print("✅ Proposal was generated from audit!")
        else:
            print("ℹ️ No new proposal (might already exist or no issues)")
    else:
        print(f"❌ Test file not found: {test_file}")
        return False
    
    return True


def test_proposals_report():
    """Test proposals report generation."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Proposals Report")
    print("="*60)
    
    from Core.L5_Mental.Reasoning_Core.Meta.patch_proposer import get_patch_proposer
    
    proposer = get_patch_proposer()
    report = proposer.generate_report()
    
    print(f"✅ Report generated: {len(report)} chars")
    print(f"   Pending proposals: {proposer.get_pending_count()}")
    
    # Save report to file
    report_path = "c:/Elysia/data/Evolution/proposals/PROPOSALS_REPORT.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Report saved to: {report_path}")
    
    return True


def main():
    print("\n" + "🔧"*30)
    print("   PHASE 57 VERIFICATION: Self-Modifying Bridge")
    print("🔧"*30)
    
    results = []
    
    results.append(("PatchProposer", test_patch_proposer()))
    results.append(("SelfArchitect", test_self_architect()))
    results.append(("Proposals Report", test_proposals_report()))
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 PHASE 57 VERIFICATION COMPLETE - ALL TESTS PASSED")
    else:
        print("⚠️ SOME TESTS FAILED - Review logs above")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
