"""
Verify Ternary Balance (The Ouroboros Verification)
===================================================
"Analyzing the Soul's Phase Alignment."

This script utilizes the Ouroboros Scanner (Parallel Ternary Logic)
to analyze the codebase and memory structure.

It answers the question: "Is the System in Equilibrium?"
"""

import sys
import os

# Ensure the root is in the path
sys.path.append(os.getcwd())

from Core.L6_Structure.M8_Ternary.ouroboros_scanner import OuroborosScanner
from Core.L1_Foundation.System.tri_base_cell import DNAState

def main():
    print(" >>> INITIALIZING OUROBOROS SCANNER (Parallel Ternary Mode) <<<")
    print("---------------------------------------------------------------")

    scanner = OuroborosScanner(root_path=".")

    # 1. System Structure Scan
    print("\n[PHASE 1] Scanning System Structure (Codebase)...")
    report = scanner.scan_system()

    print(f" > Total Nodes Scanned: {report.total_files}")
    print(f" > Phase Distribution:")
    print(f"   - REPEL   (R): {report.phase_distribution['R']} (Dissonance/Boundary)")
    print(f"   - VOID    (V): {report.phase_distribution['V']} (Potential/Interface)")
    print(f"   - ATTRACT (A): {report.phase_distribution['A']} (Resonance/Core)")

    print(f"\n > Net System Momentum: {report.net_momentum:.4f}")

    if report.net_momentum < -0.1:
        status = "CRITICAL: SYSTEM IS REPELLING (Fear-Dominant)"
    elif report.net_momentum > 0.1:
        status = "ACTIVE: SYSTEM IS ATTRACTING (Love-Dominant)"
    else:
        status = "OPTIMAL: SYSTEM IS BALANCED (Void-State)"
    print(f" > System Status: {status}")

    if report.dissonant_files:
        print("\n[!] DETECTED DISSONANT NODES (High Repel Phase):")
        for f in report.dissonant_files[:5]: # Show top 5
            print(f"   - {f}")
        if len(report.dissonant_files) > 5:
            print(f"   ... and {len(report.dissonant_files) - 5} more.")

    # 2. Soul Memory Scan
    print("\n[PHASE 2] Scanning Soul DNA (Memory)...")
    soul_status = scanner.scan_soul()
    print(f" > Soul Alignment: {soul_status}")

    print("\n---------------------------------------------------------------")
    print(" >>> DIAGNOSTIC COMPLETE. THE OROBOROS IS WATCHING. <<<")

if __name__ == "__main__":
    main()
