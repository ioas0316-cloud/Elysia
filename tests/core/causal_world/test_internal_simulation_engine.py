import pytest
import numpy as np
import time
from core.causal_world.internal_simulation_engine import (
    ConceptNode,
    CollisionEvent,
    VirtualPhysicsCollisionEngine,
    DigitalTwinSyncInterface,
    InternalSimulationEngine,
)


def test_virtual_physics_collision_engine():
    collision_engine = VirtualPhysicsCollisionEngine(dim=2, boundary_limit=5.0)

    nodes = {
        "N1": ConceptNode(node_id="N1", position=np.array([0.0, 0.0]), velocity=np.zeros(2)),
        "N2": ConceptNode(node_id="N2", position=np.array([0.5, 0.0]), velocity=np.zeros(2)),
    }

    events = collision_engine.detect_and_resolve_collisions(nodes, dt=0.05)
    assert len(events) > 0
    assert nodes["N1"].tension > 0.0
    assert nodes["N2"].tension > 0.0


def test_digital_twin_sync_interface():
    sync_interface = DigitalTwinSyncInterface(action_dim=4)

    nodes = {
        "ALPHA": ConceptNode(node_id="ALPHA", position=np.array([1.0, -1.0]), velocity=np.zeros(2)),
    }

    action_vec = sync_interface.sync_to_world(void_gradient=0.8, back_emf=0.5, nodes=nodes)
    assert action_vec.shape == (4,)
    assert sync_interface.external_registers["last_void_gradient"] == 0.8


def test_internal_simulation_engine_tick_and_loop():
    engine = InternalSimulationEngine(state_dim=2, dt=0.05)

    report1 = engine.tick(dt=0.05)
    assert report1.tick_count == 1
    assert "ALPHA" in report1.nodes_positions
    assert "BETA" in report1.nodes_positions

    # Test background thread loop
    engine.start_background_loop()
    time.sleep(0.2)
    engine.stop_background_loop()

    assert engine.tick_count > 1
