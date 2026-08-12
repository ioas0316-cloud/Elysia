import numpy as np
import pytest
from core.physics.sdf_potential_field import (
    smin,
    SDFScene,
    SDFSphere,
    SDFBox,
    SDFPlane,
    SDFPotentialFieldSimulation
)

def test_smin_blending():
    """Verify polynomial smooth minimum behavior."""
    # Test with k=0 (degenerates to simple min)
    a = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 1.5], dtype=np.float32)
    res_k0 = smin(a, b, k=0.0)
    assert np.allclose(res_k0, np.minimum(a, b))

    # Test with k > 0 (creates values smoother/lower than either input)
    # At the midpoint when a == b, smin is lower than a by k * 0.25
    a_mid = np.array([2.0], dtype=np.float32)
    b_mid = np.array([2.0], dtype=np.float32)
    k = 1.0
    res_smooth = smin(a_mid, b_mid, k=k)
    assert res_smooth[0] == pytest.approx(2.0 - k * 0.25)


def test_eikonal_property():
    """Verify the Eikonal equation property (||∇ϕ|| ≈ 1) near primitive borders."""
    sphere = SDFSphere(center=[0.0, 0.0], radius=1.0)
    scene = SDFScene()
    scene.add(sphere)

    # Sample points outside the sphere
    points = np.array([
        [1.5, 0.0],
        [0.0, -2.0],
        [1.0, 1.0]
    ], dtype=np.float32)

    grad = scene.evaluate_gradient(points)
    norms = np.linalg.norm(grad, axis=-1)

    # Gradient of SDF distance field should have unit norm
    for norm in norms:
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_box_and_plane_sdf():
    """Test box/rectangle and plane primitives."""
    # A 2D box centered at origin with half-extents of [1.0, 2.0]
    box = SDFBox(center=[0.0, 0.0], half_extents=[1.0, 2.0])

    # Point outside the right edge of box
    val_out = box.evaluate(np.array([[2.0, 0.0]], dtype=np.float32))
    assert val_out[0] == pytest.approx(1.0)

    # Point deep inside the box
    val_in = box.evaluate(np.array([[0.0, 0.0]], dtype=np.float32))
    assert val_in[0] < 0.0

    # Plane facing up (0, 1) passing through (0, 0)
    plane = SDFPlane(point=[0.0, 0.0], normal=[0.0, 1.0])
    val_plane = plane.evaluate(np.array([[5.0, 3.0], [1.0, -1.0]], dtype=np.float32))
    assert val_plane[0] == pytest.approx(3.0)
    assert val_plane[1] == pytest.approx(-1.0)


def test_huber_attraction_limits():
    """Verify that the Huber attractor potential avoids velocity explosion at large distances."""
    scene = SDFScene()
    sim = SDFPotentialFieldSimulation(scene, dimensions=2, epsilon_huber=1.0)

    target = np.array([0.0, 0.0], dtype=np.float32)

    # Near point (distance 0.1)
    p_near = np.array([[0.1, 0.0]], dtype=np.float32)
    grad_near = sim.evaluate_grad_V_attractor(p_near, target)

    # Far point (distance 1000.0)
    p_far = np.array([[1000.0, 0.0]], dtype=np.float32)
    grad_far = sim.evaluate_grad_V_attractor(p_far, target)

    # Gradient magnitude should be bounded near 1.0 even at huge distances
    assert np.linalg.norm(grad_near) < 1.0
    assert np.linalg.norm(grad_far) == pytest.approx(1.0, abs=1e-3)


def test_inward_masked_sliding():
    """Verify inward-masked sliding motion."""
    # Build a scene with a flat ground plane at y=0, normal pointing up (0, 1)
    plane = SDFPlane(point=[0.0, 0.0], normal=[0.0, 1.0])
    scene = SDFScene(blend_k=0.0)
    scene.add(plane)

    sim = SDFPotentialFieldSimulation(
        scene,
        dimensions=2,
        w_attractor=0.0,
        w_collision=0.0,
        w_light=0.0,
        collision_threshold=0.1
    )

    # Position is exactly on/near the plane (y=0.01)
    pos = np.array([[0.0, 0.01]], dtype=np.float32)
    target = np.array([0.0, 0.0], dtype=np.float32)
    light_dir = np.array([0.0, 1.0], dtype=np.float32)

    # Case 1: Velocity pointing inward (heading down, into the plane normal)
    vel_inward = np.array([[2.0, -1.0]], dtype=np.float32)
    _, next_vel_in = sim.step(pos, vel_inward, target, light_dir, dt=0.1)

    # Inward vertical component (-1.0) should be damped/projected out
    assert next_vel_in[0, 0] == pytest.approx(2.0, abs=1e-2) # horizontal sliding intact
    assert next_vel_in[0, 1] > -0.1 # inward movement suppressed

    # Case 2: Velocity pointing outward (heading up, escaping the plane)
    vel_outward = np.array([[2.0, 1.0]], dtype=np.float32)
    _, next_vel_out = sim.step(pos, vel_outward, target, light_dir, dt=0.1)

    # Outward vertical component (1.0) should remain intact (no sticking/trapping)
    assert next_vel_out[0, 0] == pytest.approx(2.0, abs=1e-2)
    assert next_vel_out[0, 1] == pytest.approx(1.0, abs=1e-2)


def test_phototaxis_flow():
    """Test phototaxis gradient pulls particles toward illuminated areas."""
    # Setup simple geometry
    sphere = SDFSphere(center=[0.0, 0.0], radius=1.0)
    scene = SDFScene()
    scene.add(sphere)

    sim = SDFPotentialFieldSimulation(scene, dimensions=2, w_light=1.0)

    # Position on top of the sphere, facing up (gradient pointing up (0,1))
    pos = np.array([[0.0, 1.01]], dtype=np.float32)
    grad_phi = scene.evaluate_gradient(pos)

    # Light direction comes from top-right (1.0, 1.0)
    light_dir = np.array([1.0, 1.0], dtype=np.float32)
    light_dir_norm = light_dir / np.linalg.norm(light_dir)

    # Evaluate light potential and its gradient
    _, grad_V_light = sim.evaluate_V_light(pos, light_dir_norm, grad_phi)

    # Force is -grad_V_light. It should push the particle towards the light direction (rightward)
    force_light = -grad_V_light[0]
    assert force_light[0] > 0.0  # Should push right, towards positive x (source of light)
