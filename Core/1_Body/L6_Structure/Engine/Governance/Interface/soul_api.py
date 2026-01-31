"""
Soul API (     )
====================

"The window through which the Observer sees the Soul."

A lightweight Flask server that exposes Elysia's internal state:
- /api/memory/nodes: Knowledge Graph nodes from HolographicMemory.
- /api/state/emotional: Emotional Spectrum from UnifiedExperienceCore.
- /api/events/recent: Recent learning events.
"""

import sys
import os
import json
import logging
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.1_Body.L2_Metabolism.Memory.unified_experience_core import UnifiedExperienceCore

app = Flask(__name__, static_folder='../../dashboard', template_folder='../../dashboard')
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SoulAPI")

# Singleton brain
core = None

def get_core():
    global core
    if core is None:
        core = UnifiedExperienceCore()
    return core

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/memory/nodes')
def get_memory_nodes():
    """Returns all concepts from TesseractMemory as graph nodes (Phase 30)."""
    try:
        from Core.1_Body.L5_Mental.Reasoning_Core.Memory.tesseract_memory import get_tesseract_memory
        memory = get_tesseract_memory()
        
        nodes = []
        edges = []
        
        for name, node in memory.nodes.items():
            # Project 4D to 3D for display
            from Core.1_Body.L5_Mental.Reasoning_Core.Topography.tesseract_geometry import TesseractGeometry
            geo = TesseractGeometry()
            x3d, y3d, z3d = geo.project_to_3d(node.vector)
            
            nodes.append({
                "id": name,
                "label": name,
                "x": x3d * 100,  # Scale for display
                "y": y3d * 100,
                "z": z3d * 100,
                "w": node.vector.w,
                "type": node.node_type,
                "amplitude": node.vector.w  # Use W as amplitude for sizing
            })
            
            # Add edges from connections
            for conn in node.connections:
                edges.append({
                    "source": name,
                    "target": conn
                })
        
        return jsonify({"nodes": nodes, "edges": edges, "stats": memory.get_stats()})
    except Exception as e:
        # Fallback to HolographicMemory
        c = get_core()
        nodes = []
        edges = []
        
        if c.holographic_memory:
            for concept_name, node in c.holographic_memory.nodes.items():
                nodes.append({
                    "id": concept_name,
                    "label": concept_name,
                    "amplitude": getattr(node, 'amplitude', 1.0),
                    "entropy": getattr(node, 'entropy', 0.5),
                    "qualia": getattr(node, 'qualia', 0.5)
                })
                connections = getattr(node, 'connections', [])
                for conn in connections:
                    edges.append({
                        "source": concept_name,
                        "target": conn
                    })
        
        return jsonify({"nodes": nodes, "edges": edges})

@app.route('/api/state/emotional')
def get_emotional_state():
    """Returns current emotional/cognitive state."""
    c = get_core()
    return jsonify({
        "aspects": c.current_state,
        "frequencies": c.aspect_frequencies
    })

@app.route('/api/events/recent')
def get_recent_events():
    """Returns recent experience events."""
    c = get_core()
    events = []
    for e in c.stream[-10:]:  # Last 10 events
        events.append({
            "id": e.id,
            "type": e.type,
            "content": e.content[:100] if e.content else "",  # Truncate
            "timestamp": e.timestamp,
            "feedback": e.feedback
        })
    return jsonify({"events": events})

if __name__ == "__main__":
    logger.info("  Soul API starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
