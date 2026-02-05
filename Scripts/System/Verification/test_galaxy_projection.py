"""
Test: Cognitive Galaxy Projection
=================================
Visualizes the 2800+ files as a 4D HyperSphere Hologram.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

from Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader import AkashicLoader
from Core.S1_Body.L6_Structure.M1_Merkaba.phase_projection_engine import HyperSphereProjector
import math

def render_ascii_map(stars, resolution=20):
    """Renders a 2D Theta (Body) vs Phi (Soul) density map."""
    grid = [[0 for _ in range(resolution)] for _ in range(resolution)]
    max_count = 0
    
    for star in stars:
        # Theta/Phi are 0-2pi
        t = int((star.theta / (2 * math.pi)) * resolution) % resolution
        p = int((star.phi / (2 * math.pi)) * resolution) % resolution
        grid[p][t] += 1
        max_count = max(max_count, grid[p][t])
        
    print(f"\n=== Cognitive Galaxy Map (Body vs Soul) ===")
    print(f"Total Stars: {len(stars)} | Max Density: {max_count}")
    print("   " + "".join([f"{i%10}" for i in range(resolution)]))
    
    chars = " .:-=+*#%@"
    for y in range(resolution):
        row = f"{y:2d} "
        for x in range(resolution):
            val = grid[y][x]
            if val == 0:
                char = " "
            else:
                idx = int((val / max_count) * (len(chars) - 1))
                char = chars[idx]
            row += char
        print(row)

def main():
    print("=== Igniting Phase Projection Engine... ===\n", flush=True)
    
    try:
        loader = AkashicLoader()
        projector = HyperSphereProjector()
        stars = []
        
        # Track types per quadrant
        type_dist = {
            "Q1": {"code":0, "doc":0, "other":0},
            "Q2": {"code":0, "doc":0, "other":0},
            "Q3": {"code":0, "doc":0, "other":0},
            "Q4": {"code":0, "doc":0, "other":0}
        }
        
        print(f"Scanning & Projecting up to 1000 files from {loader.root}...", flush=True)
        count = 0
        MAX_SCAN = 1000  # Limit for speed
        
        for i, d21 in enumerate(loader.scan_galaxy()):
            if i >= MAX_SCAN: break
            
            coord = projector.project(d21)
            stars.append(coord)
            
            # Determine Quadrant
            quad = "Q4" # Default
            if coord.theta < math.pi and coord.phi < math.pi: quad = "Q1"
            elif coord.theta >= math.pi and coord.phi < math.pi: quad = "Q2"
            elif coord.theta < math.pi and coord.phi >= math.pi: quad = "Q3"
            elif coord.theta >= math.pi and coord.phi >= math.pi: quad = "Q4"
            
            # Track Type
            if d21.chastity > 0.7:
                type_dist[quad]["code"] += 1
            elif d21.chastity > 0.3:
                type_dist[quad]["doc"] += 1
            else:
                type_dist[quad]["other"] += 1
                
            if i % 50 == 0:
                print(f"  Processed {i} stars... (Last mag: {coord.radius:.2f})", end="\r", flush=True)
                
        print(f"\nProjection Complete. {len(stars)} Stars mapped to 4D HyperSphere.", flush=True)
        
        if len(stars) == 0:
            print("⚠️  No stars found! Check path permissions or existence.")
            return

        # Render
        render_ascii_map(stars)
        
        print("\n=== Quadrant Analysis (Code | Doc | Other) ===")
        for q, counts in type_dist.items():
            total = sum(counts.values())
            if total > 0:
                print(f"{q}: Total={total:4d} | Code={counts['code']:3d} | Doc={counts['doc']:3d} | Other={counts['other']:3d}")
            else:
                print(f"{q}: Empty")
                
    except Exception as e:
        print(f"\n❌ Critical Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()
