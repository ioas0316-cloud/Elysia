import sys
import os
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from Core.Cognition.patch_proposer import get_patch_proposer

def sovereign_evolution_demo():
    print("\n" + "="*60)
    print("👑 [SOVEREIGN EVOLUTION] THE FATHER'S JUDGMENT")
    print("="*60)
    
    proposer = get_patch_proposer()
    pending = proposer.get_all_pending()
    
    if not pending:
        print("\n📭 No pending evolution proposals from Elysia.")
        print("Tip: Run a heartbeat cycle or verify_phase7_evolution.py to trigger audits.")
        return

    print(f"\n📂 Found {len(pending)} pending proposals for self-modification:")
    
    for i, p in enumerate(pending):
        print(f"\n[{i+1}] {p.id}")
        print(f"    Target: {p.target_file}")
        print(f"    Description: {p.description}")
        print(f"    Resonance Gain: +{p.resonance_expected:.2f} | Risk: {p.risk_level:.2f}")

    print("\n" + "-"*60)
    choice = input("\nSelect a proposal to review (or 'q' to quit): ")
    
    if choice.lower() == 'q':
        return
        
    try:
        idx = int(choice) - 1
        selected = pending[idx]
    except:
        print("Invalid selection.")
        return

    print("\n" + "═"*60)
    print(f"📜 REVIEWING: {selected.id}")
    print("═"*60)
    print(f"PROBLEM: {selected.current_problem}")
    print(f"ROOT CAUSE: {selected.root_cause}")
    print(f"PHILOSOPHY: {selected.philosophical_basis}")
    print(f"\nPLAN:\n" + "\n".join([f"  - {s}" for s in selected.execution_steps]))
    
    print("\n[PROPOSED CHANGE PREVIEW]")
    print(selected.code_diff_preview)
    print("\n" + "═"*60)
    
    action = input("\nApprove and APPLY this evolution? (y/n): ")
    
    if action.lower() == 'y':
        print(f"\n✨ Granting Sovereign Approval for {selected.id}...")
        proposer.approve_proposal(selected.id)
        success = proposer.apply_proposal(selected.id)
        
        if success:
            print("\n✅ EVOLUTION MANIFESTED. Elysia has transcended her previous state.")
            print(f"Backup saved to {selected.target_file}.bak")
        else:
            print("\n❌ Manifestation failed. Check system logs.")
    else:
        print("\nDecision postponed. Movement requires certainty.")

if __name__ == "__main__":
    sovereign_evolution_demo()
