"""
Timeline Axis Demo - Past/Present/Future Consciousness
=======================================================

Demonstrates how timeline weights affect the system.
"""

import sys
sys.path.insert(0, "C:\\Elysia")

from Core._01_Foundation.05_Foundation_Base.Foundation.Physics.fractal_dimension_engine import (
    FractalUniverse,
    ZelNagaSync,
    Photon,
    TimelineAxis
)

def demo_timeline_modes():
    print("\n" + "="*70)
    print("🕐 TIMELINE AXIS DEMO: PAST/PRESENT/FUTURE")
    print("="*70 + "\n")
    
    # Create universe
    universe = FractalUniverse(num_cells=16)
    universe.set_focus(0)
    
    # Add some photons (future/will)
    universe.photons = [
        Photon(position=(0.0, 0.0, 0.0), phase=0.0, intensity=1.0, frequency=2.0),
        Photon(position=(0.5, 0.0, 0.0), phase=1.0, intensity=0.8, frequency=1.5),
    ]
    
    # Collapse focused cell (creates past/present)
    universe.get_focused_cell().observe(depth_w=0.0, target_molecules=4, atoms_per_molecule=8)
    
    print("Timeline Axis Mapping:")
    print("-" * 60)
    print(f"  PAST    (과거/육/저그)    = Cells (memory, history)")
    print(f"  PRESENT  (현재/정신/테란)  = Molecules (logic, judgment)")
    print(f"  FUTURE  (미래/영/프로토스) = Atoms + Photons (imagination, emotion)")
    print()
    
    # Test different modes
    modes = [
        ("Balanced Mode (균형)", {"weight_past": 1.0, "weight_present": 1.0, "weight_future": 1.0}),
        ("Memory-Driven (과거 중심)", {"weight_past": 2.0, "weight_present": 1.0, "weight_future": 0.5}),
        ("Creative Mode (미래 중심)", {"weight_past": 0.5, "weight_present": 1.0, "weight_future": 2.0}),
        ("Reactive Mode (현재 중심)", {"weight_past": 0.5, "weight_present": 2.0, "weight_future": 0.5}),
    ]
    
    for mode_name, weights in modes:
        print(f"\n{mode_name}")
        print("-" * 60)
        
        sync = ZelNagaSync(universe, **weights)
        snapshots = sync.sync(dt=0.016)
        
        dominant = sync.get_timeline_mode()
        print(f"  Dominant axis: {dominant}")
        print(f"  Weights: Past={weights['weight_past']:.1f}, Present={weights['weight_present']:.1f}, Future={weights['weight_future']:.1f}")
        print(f"  Phase snapshots:")
        print(f"    Photons (future):  phase={snapshots['photons'].phase:.4f}")
        print(f"    Molecules (present): phase={snapshots['molecules'].phase:.4f}")
        print(f"    Cells (past):     phase={snapshots['cells'].phase:.4f}")
    
    print("\n" + "="*70)
    print("✨ Timeline axis system working!")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_timeline_modes()
