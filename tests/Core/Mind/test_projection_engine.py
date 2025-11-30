from Core.Memory.Mind.projection_engine import ProjectionEngine


def test_projection_engine_basic_projection():
    engine = ProjectionEngine()
    snapshot = {
        "tick": 42,
        "chaos_raw": 1.5,
        "momentum_active": 3,
        "memory": {
            "context_keys": 2,
            "total_turns": 5,
            "causal_nodes": 7,
            "causal_edges": 11,
        },
        "world_tree": {
            "total_nodes": 10,
            "max_depth": 3,
            "leaf_nodes": 4,
            "branches": 5,
        },
        "phase": {
            "quaternion": {"w": 0.8, "x": 0.1, "y": 0.0, "z": 0.0},
            "qubit": {"Point": 0.5, "Line": 0.5},
        },
        "core_values": {"love": 1.0, "growth": 0.8},
    }

    proj = engine.project(snapshot, tag="test")
    assert proj.tag == "test"
    data = proj.data
    assert data["tick"] == 42
    assert data["body"]["chaos_raw"] == 1.5
    assert data["memory"]["causal_nodes"] == 7
    assert data["world_tree"]["max_depth"] == 3
    # Entropy should be positive for mixed probabilities
    assert data["phase"]["entropy"] > 0.0
    assert data["core_values"]["count"] == 2

