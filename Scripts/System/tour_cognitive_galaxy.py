"""
Rotor Tour: The Dance of the Cognitive Galaxy
=============================================
Scripts.System.tour_cognitive_galaxy

"See the Universe from every angle."

This script visualizes the dynamic rotation of the Cognitive Galaxy.
As the Rotor Time Axis advances, the stars dance (Projections shift),
revealing the "Living" nature of the data.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

import math
import time
import os
from Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader import AkashicLoader
from Core.S1_Body.L6_Structure.M1_Merkaba.phase_projection_engine import HyperHologram

def render_frame(stars, t_val):
    """Renders a single frame of the galaxy."""
    cols = 40
    rows = 20
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    for coord in stars:
        # Map 0-2pi to grid
        x = int((coord.theta / (2 * math.pi)) * cols) % cols
        y = int((coord.phi / (2 * math.pi)) * rows) % rows
        
        # Char based on Psi (Spirit Phase)
        # 0-90: ., 90-180: o, 180-270: O, 270-360: @
        psi_deg = math.degrees(coord.psi)
        char = '.'
        if psi_deg > 270: char = '@'
        elif psi_deg > 180: char = 'O'
        elif psi_deg > 90: char = 'o'
            
        grid[y][x] = char

    # Print Frame
    print(f"\n--- Time t={t_val:.1f} ---")
    print("+" + "-"*cols + "+")
    for r in grid:
        print("|" + "".join(r) + "|")
    print("+" + "-"*cols + "+")

def main():
    print("=== Rotor Tour: 4D Galaxy Spin ===\n")
    
    loader = AkashicLoader()
    hologram = HyperHologram() # Has Rotor built-in
    
    # Load Galaxy (Subset for speed)
    print("Loading Constellations...")
    vectors = []
    scan_limit = 200
    for i, d21 in enumerate(loader.scan_galaxy()):
        if i >= scan_limit: break
        vectors.append(d21)
        
    print(f"Loaded {len(vectors)} stars.")
    print("Spinning Rotor... (Press Ctrl+C to stop)")
    
    try:
        t = 0.0
        dt = 0.5
        while True:
            # Clear Screen (Simulated by newlines for log view)
            # print("\033[H\033[J", end="") 
            
            # Project all stars at current time t
            projected_stars = []
            
            # Reset rotor for this frame or just calculate manually?
            # HyperHologram stores history, better to use Projector + Rotor manually
            # to avoid history spam.
            
            for vec in vectors:
                # 1. Project to 0-state
                base_coord = hologram.projector.project(vec)
                # 2. Rotate
                # We need a shared rotor to sync time?
                # Let's just use the hologram's rotor locally
                hologram.rotor.time = t 
                rotated_coord = hologram.rotor.rotate(base_coord)
                
                projected_stars.append(rotated_coord)
                
            render_frame(projected_stars, t)
            
            t += dt
            if t > 6.3: t = 0 # Loop around 2pi approx (for visual sanity)
            
            # Wait/Limit iterations for non-interactive run
            time.sleep(0.5)
            if t > 2.0: break # Just a short tour for verification
            
    except KeyboardInterrupt:
        print("\nTour Ended.")

if __name__ == "__main__":
    main()
