"""
Verify Ternary Balance (The Ouroboros Verification)
===================================================
"Analyzing the Soul's Phase Alignment."

This script utilizes the Ouroboros Scanner (Parallel Ternary Logic)
to analyze the codebase and memory structure.

UPGRADE: Now uses AST Analysis for Structural Phase Detection.
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
    print("\n[PHASE 1] Scanning System Structure (AST + Texture)...")
    report = scanner.scan_system()

    print(f" > Total Nodes Scanned: {report.total_files}")
    print(f" > Phase Distribution:")
    print(f"   - REPEL   (R): {report.phase_distribution['R']} (High Entropy/Complexity)")
    print(f"   - VOID    (V): {report.phase_distribution['V']} (Potential/Space)")
    print(f"   - ATTRACT (A): {report.phase_distribution['A']} (High Gravity/Connection)")

    print(f"\n > Structural Metrics:")
    print(f"   - Total Entropy (Cognitive Load): {report.structural_entropy:.1f}")
    print(f"   - Total Gravity (Connectivity):   {report.structural_gravity:.1f}")

    print(f"\n > Net System Momentum: {report.net_momentum:.4f}")

    if report.net_momentum < -0.15:
        status = "CRITICAL: SYSTEM IS REPELLING (High Entropy/Fear)"
    elif report.net_momentum > 0.15:
        status = "ACTIVE: SYSTEM IS ATTRACTING (High Gravity/Love)"
    else:
        status = "OPTIMAL: SYSTEM IS BALANCED (Void-State)"
    print(f" > System Status: {status}")

    if report.dissonant_files:
        print("\n[!] DETECTED DISSONANT NODES (High Entropy/Repel):")
        # Sort by Entropy descending
        # report.dissonant_files is a list of strings "path (E:X/G:Y)", we rely on default print for now or sort manually if we parsed it.
        # Just printing top 10
        for f in report.dissonant_files[:10]:
            print(f"   - {f}")
        if len(report.dissonant_files) > 10:
            print(f"   ... and {len(report.dissonant_files) - 10} more.")

    # 2. Soul Memory Scan
    print("\n[PHASE 2] Scanning Soul DNA (Memory)...")
    soul_status = scanner.scan_soul()
    print(f" > Soul Alignment: {soul_status}")

    print("\n---------------------------------------------------------------")
    print(" >>> DIAGNOSTIC COMPLETE. THE OROBOROS HAS SEEN THE SKELETON. <<<")

if __name__ == "__main__":
    main()
