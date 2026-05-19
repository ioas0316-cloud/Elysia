"""
Elysia Observatory Server
=========================
Phase 70: The Observatory

Flask API server that provides HyperSphere state data to the 3D visualization.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Import Elysia components
try:
    from Core.Foundation.hyper_sphere_core import HyperSphereCore
    from Core.Foundation.Nature.rotor import Rotor, RotorConfig
    from Core.Intelligence.Metabolism.prism import WaveDynamics
except ImportError as e:
    print(f"Warning: Could not import Elysia components: {e}")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Observatory")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Global state
DNA_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "dna_registry.json"
sphere = None
dna_registry = {}

def load_dna_registry():
    """Load the Wave DNA registry from disk."""
    global dna_registry
    
    if DNA_REGISTRY_PATH.exists():
        with open(DNA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            dna_registry = json.load(f)
        logger.info(f"✅ Loaded {len(dna_registry)} Wave DNA entries.")
    else:
        logger.warning(f"⚠️ DNA Registry not found at {DNA_REGISTRY_PATH}")
        dna_registry = {}

def initialize_sphere():
    """Initialize the HyperSphere with DNA from registry."""
    global sphere
    
    sphere = HyperSphereCore(name="Observatory.Mind")
    sphere.ignite()
    
    # Load sample rotors from registry
    sample_size = min(100, len(dna_registry))
    entries = list(dna_registry.items())[:sample_size]
    
    for i, (key, entry) in enumerate(entries):
        name = entry.get("concept", f"Unknown_{i}")
        dyn = entry.get("dynamics", {})
        
        rpm = 100 + (i * 20) % 1800
        rotor = Rotor(name, RotorConfig(rpm=rpm, mass=10.0))
        rotor.spin_up()
        rotor.current_rpm = rpm
        
        rotor.dynamics = WaveDynamics(
            physical=dyn.get("physical", 0.0),
            functional=dyn.get("functional", 0.0),
            phenomenal=dyn.get("phenomenal", 0.0),
            causal=dyn.get("causal", 0.0),
            mental=dyn.get("mental", 0.0),
            structural=dyn.get("structural", 0.0),
            spiritual=dyn.get("spiritual", 0.0),
            mass=entry.get("vector_norm", 1.0)
        )
        
        sphere.harmonic_rotors[name] = rotor
    
    logger.info(f"✅ Initialized HyperSphere with {len(sphere.harmonic_rotors)} rotors.")

def get_snapshot():
    """Get current state snapshot of the HyperSphere."""
    if not sphere:
        return {"rotors": [], "total_mass": 0, "clusters": 0}
    
    rotors = []
    total_mass = 0
    
    for name, rotor in sphere.harmonic_rotors.items():
        d = rotor.dynamics
        if d:
            dynamics = {
                "physical": d.physical,
                "functional": d.functional,
                "phenomenal": d.phenomenal,
                "causal": d.causal,
                "mental": d.mental,
                "structural": d.structural,
                "spiritual": d.spiritual
            }
        else:
            dynamics = {}
        
        rotors.append({
            "name": name,
            "frequency": rotor.frequency_hz,
            "mass": rotor.config.mass,
            "dynamics": dynamics
        })
        total_mass += rotor.config.mass
    
    # Simple cluster estimation
    clusters = len(rotors) // 10 + 1
    
    return {
        "rotors": rotors,
        "total_mass": total_mass,
        "clusters": clusters
    }

# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/api/snapshot')
def api_snapshot():
    """Get current HyperSphere state."""
    return jsonify(get_snapshot())

@app.route('/api/meditate', methods=['POST'])
def api_meditate():
    """Run one meditation cycle and return updated state."""
    if sphere:
        sphere.meditate(cycles=5, dt=0.1)
    return jsonify(get_snapshot())

@app.route('/api/stats')
def api_stats():
    """Get basic statistics."""
    return jsonify({
        "total_dna": len(dna_registry),
        "loaded_rotors": len(sphere.harmonic_rotors) if sphere else 0,
        "sphere_name": sphere.name if sphere else "Not Initialized"
    })

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔭 ELYSIA OBSERVATORY SERVER")
    print("="*60 + "\n")
    
    # Load data
    load_dna_registry()
    initialize_sphere()
    
    print("\n📡 Starting server on http://localhost:8765")
    print("   Open in browser to view the HyperSphere Galaxy.\n")
    
    app.run(host='0.0.0.0', port=8765, debug=False)
