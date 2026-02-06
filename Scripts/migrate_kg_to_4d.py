"""
Migration Script: Convert 3D KG to 4D HyperSphere
=================================================
Converts existing kg_with_embeddings.json from:
  position: {x, y, z}
to:
  hypersphere: {theta, phi, psi, r}

Using spherical coordinate conversion:
  r = sqrt(x^2 + y^2 + z^2)
  theta = atan2(y, x)
  phi = acos(z / r) if r > 0 else 0
  psi = 0 (4th dimension, starts at origin)
"""

import json
import math
from pathlib import Path

KG_PATH = Path("data/kg_with_embeddings.json")
BACKUP_PATH = Path("data/kg_with_embeddings_3d_backup.json")

def cartesian_to_hypersphere(pos):
    """Convert 3D Cartesian to 4D HyperSphere coordinates."""
    x = pos.get('x', 0)
    y = pos.get('y', 0)
    z = pos.get('z', 0)
    
    r = math.sqrt(x**2 + y**2 + z**2)
    
    if r > 0:
        theta = math.atan2(y, x)  # Angle in xy-plane
        phi = math.acos(z / r)    # Angle from z-axis
    else:
        theta = 0.0
        phi = 0.0
    
    psi = 0.0  # 4th dimension starts at 0
    
    return {
        "theta": round(theta, 6),
        "phi": round(phi, 6),
        "psi": round(psi, 6),
        "r": round(r, 6)
    }

def migrate():
    print("Loading KG from:", KG_PATH)
    
    with open(KG_PATH, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    # Backup original
    print("Creating backup at:", BACKUP_PATH)
    with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    
    # Migrate nodes
    migrated = 0
    for node in kg.get('nodes', []):
        if 'position' in node and 'hypersphere' not in node:
            node['hypersphere'] = cartesian_to_hypersphere(node['position'])
            del node['position']
            migrated += 1
    
    # Save migrated KG
    print(f"Migrated {migrated} nodes to 4D HyperSphere format")
    with open(KG_PATH, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
