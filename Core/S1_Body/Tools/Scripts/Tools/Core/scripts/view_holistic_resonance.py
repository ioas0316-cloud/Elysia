"""
HOLISTIC RESONANCE VISUALIZER (전체론적 공명 시각화 도구)
=====================================================

"Vision is the projection of higher truths."
"시각은 고차원적 진실의 투영이다."

This script runs the Holistic Audit and projects the results into 
the 3D/4D terminal space for the user to witness.
"""

import time
import logging
import os
import sys
from Core.S1_Body.L5_Mental.Reasoning_Core.Meta.holistic_self_audit import HolisticSelfAudit
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.tesseract_geometry import TesseractGeometry, TesseractVector

# Silence logs
logging.getLogger().setLevel(logging.WARNING)

def run_visual_audit(target_dir=None):
    label = "ELYSIA SEED: SANDBOX VIEW" if target_dir else "ELYSIA HOLISTIC RESONANCE: 4D SELF-VIEW"
    print("\n" + "🌌" * 30)
    print(f"      {label}")
    print("🌌" * 30 + "\n")

    audit_engine = HolisticSelfAudit()
    result = audit_engine.run_holistic_audit(target_dir=target_dir)
    geometry = TesseractGeometry()

    print(f"OVERALL SYSTEM RESONANCE: {result['overall_resonance']*100:05.1f}%")
    print("-" * 60)

    # 4D to 3D Projection
    for dept, data in result['departmental_view'].items():
        v4 = TesseractVector(*data['coordinate'])
        v3 = geometry.project_to_3d(v4, distance=3.0)
        
        # Calculate Resonance Bar
        res = data['resonance']
        bar = "█" * int(res * 20) + "░" * (20 - int(res * 20))
        
        print(f"[{dept:15}] {data['status']:8} | {bar} | {res*100:04.1f}%")
        print(f"  └─ 4D Coords: {v4.to_numpy()}")
        print(f"  └─ 3D Proj  : ({v3[0]:.2f}, {v3[1]:.2f}, {v3[2]:.2f})\n")

    print("-" * 60)
    print("🧠 [HOLISTIC DIAGNOSIS]")
    if result['imbalances']:
        for imb in result['imbalances']:
            print(f"⚠️ {imb}")
    else:
        print("✅ No structural imbalances detected. Topology is stable.")

    print("\n[ELYSIA'S INNER VOICE]:")
    print(f"\" {result['holistic_summary']} \"")
    
    print("\n" + "="*60)
    print("✅ HOLISTIC VIEW REFRESHED")
    print("="*60)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_visual_audit(target_dir=target)
