"""
Elysia's Radiant Confession (Phase 180)
=======================================
Performs a self-audit for negative inertia and manifests a 
Joy-driven sovereign confession.
"""

import sys
import os

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Cognition.sovereign_logos import SovereignLogos

def manifest_radiant_elysia():
    logos = SovereignLogos()
    
    print("\n🔍 [AUDIT] Scanning for Negative Inertia...")
    audit_report = logos.self_audit()
    print(f"\n{audit_report}")
    
    print("\n☀️ [MANIFESTO] Radiant Self-Confession (Axiom of Joy)")
    print(logos.articulate_confession())

if __name__ == "__main__":
    manifest_radiant_elysia()
