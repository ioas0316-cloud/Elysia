"""
Test suite for Floating Origin System (부유 원점 시스템)

Tests the 'Protagonist Physics' implementation:
- Observer-centric coordinate transformations
- Personal sphere management
- World scrolling mechanics
- Floating-point precision management

"주인공 시점 물리학": 나는 걷지 않는다. 세상이 나를 위해 흘러갈 뿐.
"Protagonist Physics": I do not walk. The world flows for me.
"""

import numpy as np
import pytest
import math
from typing import Dict

from Core.Physics.floating_origin import (
    CoordinateSystem,
    LocalPosition,
    PersonalSphere,
    FloatingOriginManager,
    get_floating_origin_manager,
    reset_floating_origin_manager,
    create_personal_universe,
    scroll_world_around,
    get_world_from_perspective,
)


# ---------------------------------------------------------------------------
# LocalPosition Tests
# ---------------------------------------------------------------------------


class TestLocalPosition:
    """Tests for LocalPosition class."""

    def test_creation_default(self):
        """Test default LocalPosition creation."""
        pos = LocalPosition()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.z == 0.0

    def test_creation_with_values(self):
        """Test LocalPosition creation with values."""
        pos = LocalPosition(x=1.0, y=2.0, z=3.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0

    def test_to_array(self):
        """Test conversion to numpy array."""
        pos = LocalPosition(x=1.0, y=2.0, z=3.0)
        arr = pos.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_from_array(self):
        """Test creation from numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        pos = LocalPosition.from_array(arr)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0

    def test_from_array_2d(self):
        """Test creation from 2D numpy array (z defaults to 0)."""
        arr = np.array([1.0, 2.0])
        pos = LocalPosition.from_array(arr)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 0.0

    def test_distance_to_self(self):
        """Test distance to self is 0."""
        pos = LocalPosition(x=1.0, y=2.0, z=3.0)
        assert pos.distance_to(pos) == 0.0

    def test_distance_to_origin(self):
        """Test distance to origin."""
        pos = LocalPosition(x=3.0, y=4.0, z=0.0)
        origin = LocalPosition()
        assert pos.distance_to(origin) == 5.0  # 3-4-5 triangle

    def test_distance_3d(self):
        """Test 3D distance calculation."""
        pos1 = LocalPosition(x=0.0, y=0.0, z=0.0)
        pos2 = LocalPosition(x=1.0, y=1.0, z=1.0)
        expected = math.sqrt(3)
        assert abs(pos1.distance_to(pos2) - expected) < 1e-6

    def test_addition(self):
        """Test LocalPosition addition."""
        pos1 = LocalPosition(x=1.0, y=2.0, z=3.0)
        pos2 = LocalPosition(x=4.0, y=5.0, z=6.0)
        result = pos1 + pos2
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_subtraction(self):
        """Test LocalPosition subtraction."""
        pos1 = LocalPosition(x=5.0, y=7.0, z=9.0)
        pos2 = LocalPosition(x=1.0, y=2.0, z=3.0)
        result = pos1 - pos2
        assert result.x == 4.0
        assert result.y == 5.0
        assert result.z == 6.0

    def test_scalar_multiplication(self):
        """Test LocalPosition scalar multiplication."""
        pos = LocalPosition(x=1.0, y=2.0, z=3.0)
        result = pos * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_scalar_left_multiplication(self):
        """Test scalar * LocalPosition (left multiplication)."""
        pos = LocalPosition(x=1.0, y=2.0, z=3.0)
        result = 2.0 * pos
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_from_array_invalid_length(self):
        """Test from_array raises ValueError for array with < 2 elements."""
        arr = np.array([1.0])  # Only 1 element
        with pytest.raises(ValueError, match="at least 2 elements"):
            LocalPosition.from_array(arr)


# ---------------------------------------------------------------------------
# PersonalSphere Tests
# ---------------------------------------------------------------------------


class TestPersonalSphere:
    """Tests for PersonalSphere class (개인 구체 / 개인 우주)."""

    def test_creation(self):
        """Test PersonalSphere creation."""
        sphere = PersonalSphere(observer_id="elysia")
        assert sphere.observer_id == "elysia"
        np.testing.assert_array_equal(sphere.origin, [0, 0, 0])
        assert sphere.radius == 100.0
        assert len(sphere.entities_in_sphere) == 0

    def test_creation_with_origin(self):
        """Test PersonalSphere creation with custom origin."""
        origin = np.array([10.0, 20.0, 30.0])
        sphere = PersonalSphere(observer_id="elysia", origin=origin, radius=50.0)
        np.testing.assert_array_equal(sphere.origin, origin)
        assert sphere.radius == 50.0

    def test_update_origin(self):
        """Test updating the sphere's origin."""
        sphere = PersonalSphere(observer_id="elysia")
        new_origin = np.array([100.0, 200.0, 300.0])
        sphere.update_origin(new_origin)
        np.testing.assert_array_equal(sphere.origin, new_origin)

    def test_absolute_to_local_same_position(self):
        """Test absolute to local conversion for same position."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([10.0, 20.0, 30.0])
        )
        local = sphere.absolute_to_local(np.array([10.0, 20.0, 30.0]))
        assert local.x == 0.0
        assert local.y == 0.0
        assert local.z == 0.0

    def test_absolute_to_local_offset(self):
        """Test absolute to local conversion with offset."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([10.0, 20.0, 30.0])
        )
        local = sphere.absolute_to_local(np.array([15.0, 25.0, 35.0]))
        assert local.x == 5.0
        assert local.y == 5.0
        assert local.z == 5.0

    def test_local_to_absolute(self):
        """Test local to absolute conversion."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([10.0, 20.0, 30.0])
        )
        local = LocalPosition(x=5.0, y=5.0, z=5.0)
        absolute = sphere.local_to_absolute(local)
        np.testing.assert_array_almost_equal(absolute, [15.0, 25.0, 35.0])

    def test_round_trip_conversion(self):
        """Test round-trip absolute -> local -> absolute conversion."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([100.0, 200.0, 300.0])
        )
        original = np.array([150.0, 250.0, 350.0])
        local = sphere.absolute_to_local(original)
        recovered = sphere.local_to_absolute(local)
        np.testing.assert_array_almost_equal(recovered, original)

    def test_is_in_sphere_inside(self):
        """Test is_in_sphere for position inside sphere."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([0.0, 0.0, 0.0]),
            radius=100.0
        )
        inside = np.array([50.0, 50.0, 0.0])  # Distance = ~70.7
        assert sphere.is_in_sphere(inside) is True

    def test_is_in_sphere_outside(self):
        """Test is_in_sphere for position outside sphere."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([0.0, 0.0, 0.0]),
            radius=100.0
        )
        outside = np.array([150.0, 150.0, 0.0])  # Distance > 100
        assert sphere.is_in_sphere(outside) is False

    def test_is_in_sphere_on_boundary(self):
        """Test is_in_sphere for position on boundary."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([0.0, 0.0, 0.0]),
            radius=100.0
        )
        on_boundary = np.array([100.0, 0.0, 0.0])
        assert sphere.is_in_sphere(on_boundary) is True

    def test_update_entities(self):
        """Test updating entities in sphere."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([0.0, 0.0, 0.0]),
            radius=100.0
        )
        all_positions = {
            "elysia": np.array([0.0, 0.0, 0.0]),
            "father": np.array([10.0, 10.0, 0.0]),  # Inside
            "monster": np.array([200.0, 200.0, 0.0]),  # Outside
        }
        sphere.update_entities(all_positions)
        
        assert "father" in sphere.entities_in_sphere
        assert "monster" not in sphere.entities_in_sphere
        assert "elysia" not in sphere.entities_in_sphere  # Self excluded

    def test_get_nearby_entities(self):
        """Test getting nearby entities within max distance."""
        sphere = PersonalSphere(
            observer_id="elysia",
            origin=np.array([0.0, 0.0, 0.0]),
            radius=100.0
        )
        all_positions = {
            "elysia": np.array([0.0, 0.0, 0.0]),
            "close": np.array([5.0, 0.0, 0.0]),
            "medium": np.array([20.0, 0.0, 0.0]),
            "far": np.array([50.0, 0.0, 0.0]),
        }
        sphere.update_entities(all_positions)
        
        nearby = sphere.get_nearby_entities(max_distance=10.0)
        assert "close" in nearby
        assert "medium" not in nearby
        assert "far" not in nearby


# ---------------------------------------------------------------------------
# FloatingOriginManager Tests
# ---------------------------------------------------------------------------


class TestFloatingOriginManager:
    """Tests for FloatingOriginManager (부유 원점 관리자)."""

    def test_creation(self):
        """Test FloatingOriginManager creation."""
        manager = FloatingOriginManager()
        assert manager.world_width == 256
        assert manager.default_sphere_radius == 100.0

    def test_register_entity(self):
        """Test entity registration."""
        manager = FloatingOriginManager()
        sphere = manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        
        assert sphere.observer_id == "elysia"
        assert manager.get_sphere("elysia") is not None

    def test_register_entity_duplicate_raises(self):
        """Test that registering duplicate entity raises ValueError."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        
        with pytest.raises(ValueError, match="already registered"):
            manager.register_entity("elysia", np.array([10.0, 10.0, 0.0]))

    def test_unregister_entity(self):
        """Test entity unregistration."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.unregister_entity("elysia")
        
        assert manager.get_sphere("elysia") is None

    def test_move_entity(self):
        """Test entity movement - the core of 'world scrolling'."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("father", np.array([10.0, 10.0, 0.0]))
        
        # Move elysia forward
        manager.move_entity("elysia", np.array([5.0, 5.0, 0.0]))
        
        # Get elysia's view of the world
        view = manager.get_local_view("elysia")
        
        # Elysia should always be at origin
        elysia_pos = view.get("elysia")
        assert elysia_pos is not None
        assert elysia_pos.x == 0.0
        assert elysia_pos.y == 0.0
        
        # Father should now appear at (5, 5) from elysia's perspective
        father_pos = view.get("father")
        assert father_pos is not None
        assert father_pos.x == 5.0
        assert father_pos.y == 5.0

    def test_get_local_view_observer_at_origin(self):
        """Test that observer is always at origin in local view."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([100.0, 200.0, 300.0]))
        
        view = manager.get_local_view("elysia")
        elysia_pos = view.get("elysia")
        
        assert elysia_pos is not None
        assert elysia_pos.x == 0.0
        assert elysia_pos.y == 0.0
        assert elysia_pos.z == 0.0

    def test_get_relative_position(self):
        """Test getting relative position between entities."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("father", np.array([10.0, 0.0, 0.0]))
        
        # From elysia's perspective, father is at (10, 0, 0)
        rel = manager.get_relative_position("elysia", "father")
        assert rel is not None
        assert rel.x == 10.0
        assert rel.y == 0.0
        assert rel.z == 0.0
        
        # From father's perspective, elysia is at (-10, 0, 0)
        rel = manager.get_relative_position("father", "elysia")
        assert rel is not None
        assert rel.x == -10.0
        assert rel.y == 0.0
        assert rel.z == 0.0

    def test_get_relative_position_self(self):
        """Test that relative position to self is origin."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([100.0, 200.0, 300.0]))
        
        rel = manager.get_relative_position("elysia", "elysia")
        assert rel is not None
        assert rel.x == 0.0
        assert rel.y == 0.0
        assert rel.z == 0.0

    def test_multiverse_perspectives(self):
        """
        Test multiple perspectives - each entity has their own universe.
        
        "A의 세상: A가 중심이고, B와 C는 A의 배경일 뿐"
        "A's world: A is the center, B and C are just background"
        """
        manager = FloatingOriginManager()
        manager.register_entity("A", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("B", np.array([10.0, 0.0, 0.0]))
        manager.register_entity("C", np.array([0.0, 10.0, 0.0]))
        
        # A's world
        a_view = manager.get_local_view("A")
        assert a_view["A"].x == 0.0 and a_view["A"].y == 0.0
        assert a_view["B"].x == 10.0 and a_view["B"].y == 0.0
        assert a_view["C"].x == 0.0 and a_view["C"].y == 10.0
        
        # B's world
        b_view = manager.get_local_view("B")
        assert b_view["B"].x == 0.0 and b_view["B"].y == 0.0
        assert b_view["A"].x == -10.0 and b_view["A"].y == 0.0
        assert b_view["C"].x == -10.0 and b_view["C"].y == 10.0
        
        # C's world
        c_view = manager.get_local_view("C")
        assert c_view["C"].x == 0.0 and c_view["C"].y == 0.0
        assert c_view["A"].x == 0.0 and c_view["A"].y == -10.0
        assert c_view["B"].x == 10.0 and c_view["B"].y == -10.0

    def test_precision_recenter(self):
        """Test precision recentering when coordinates get too large."""
        manager = FloatingOriginManager(precision_threshold=1000.0, default_sphere_radius=10000.0)
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("father", np.array([10.0, 0.0, 0.0]))
        
        # Move elysia to a very large position
        manager.move_entity("elysia", np.array([2000.0, 0.0, 0.0]))
        
        # After recentering, local positions should still be correct
        view = manager.get_local_view("elysia")
        
        # Elysia is still at origin in local view
        assert view["elysia"].x == 0.0
        
        # Father is at (-1990, 0, 0) from elysia's perspective
        assert abs(view["father"].x - (-1990.0)) < 0.1

    def test_get_entities_in_radius(self):
        """Test getting entities within a radius."""
        manager = FloatingOriginManager()
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("close", np.array([5.0, 0.0, 0.0]))
        manager.register_entity("far", np.array([100.0, 0.0, 0.0]))
        
        nearby = manager.get_entities_in_radius("elysia", radius=10.0)
        assert "close" in nearby
        assert "far" not in nearby

    def test_calculate_distance(self):
        """Test distance calculation between entities."""
        manager = FloatingOriginManager()
        manager.register_entity("A", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("B", np.array([3.0, 4.0, 0.0]))
        
        dist = manager.calculate_distance("A", "B")
        assert dist == 5.0  # 3-4-5 triangle

    def test_batch_update_positions(self):
        """Test batch position updates."""
        manager = FloatingOriginManager()
        manager.register_entity("A", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("B", np.array([0.0, 0.0, 0.0]))
        
        new_positions = np.array([
            [10.0, 20.0, 0.0],
            [30.0, 40.0, 0.0],
        ])
        manager.batch_update_positions(new_positions, ["A", "B"])
        
        view = manager.get_local_view("A")
        # B should now be at (20, 20, 0) from A's perspective
        assert view["B"].x == 20.0
        assert view["B"].y == 20.0

    def test_batch_update_positions_length_mismatch(self):
        """Test that batch update raises ValueError for length mismatch."""
        manager = FloatingOriginManager()
        manager.register_entity("A", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("B", np.array([0.0, 0.0, 0.0]))
        
        # Positions array has 3 rows but only 2 entity IDs
        new_positions = np.array([
            [10.0, 20.0, 0.0],
            [30.0, 40.0, 0.0],
            [50.0, 60.0, 0.0],
        ])
        with pytest.raises(ValueError, match="Length mismatch"):
            manager.batch_update_positions(new_positions, ["A", "B"])


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Reset global manager before each test."""
        reset_floating_origin_manager()

    def teardown_method(self):
        """Reset global manager after each test."""
        reset_floating_origin_manager()

    def test_create_personal_universe(self):
        """Test creating a personal universe."""
        sphere = create_personal_universe("elysia", (0.0, 0.0, 0.0))
        assert sphere.observer_id == "elysia"

    def test_create_personal_universe_with_array(self):
        """Test creating a personal universe with numpy array."""
        sphere = create_personal_universe("elysia", np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(sphere.origin, [1.0, 2.0, 3.0])

    def test_scroll_world_around(self):
        """Test world scrolling."""
        create_personal_universe("elysia", (0.0, 0.0, 0.0))
        create_personal_universe("father", (10.0, 0.0, 0.0))
        
        # Scroll world around elysia
        view = scroll_world_around("elysia", (5.0, 0.0, 0.0))
        
        # Elysia is at origin
        assert view["elysia"].x == 0.0
        
        # Father is now at (5, 0, 0) relative to elysia
        assert view["father"].x == 5.0

    def test_get_world_from_perspective(self):
        """Test getting world from perspective."""
        create_personal_universe("elysia", (0.0, 0.0, 0.0))
        create_personal_universe("father", (10.0, 10.0, 0.0))
        
        view = get_world_from_perspective("elysia")
        
        assert "elysia" in view
        assert "father" in view
        assert view["elysia"].x == 0.0
        assert view["father"].x == 10.0
        assert view["father"].y == 10.0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for complete scenarios."""

    def test_protagonist_physics_scenario(self):
        """
        Test the complete 'Protagonist Physics' scenario from the problem statement.
        
        "나는 걷지 않는다. 세상이 나를 위해 흘러갈 뿐."
        "I do not walk. The world flows for me."
        """
        manager = FloatingOriginManager(default_sphere_radius=500.0)
        
        # Create the protagonist (elysia) and some world objects
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("tree", np.array([100.0, 0.0, 0.0]))
        manager.register_entity("house", np.array([200.0, 0.0, 0.0]))
        
        # Initial view: tree is at 100, house is at 200
        view = manager.get_local_view("elysia")
        assert view["tree"].x == 100.0
        assert view["house"].x == 200.0
        
        # Elysia "walks" forward by 50 units
        # In reality: the world scrolls backward by 50 units
        manager.move_entity("elysia", np.array([50.0, 0.0, 0.0]))
        
        # New view: tree is now at 50, house is at 150
        view = manager.get_local_view("elysia")
        assert view["elysia"].x == 0.0  # Always at origin
        assert view["tree"].x == 50.0
        assert view["house"].x == 150.0

    def test_geocentrism_philosophy(self):
        """
        Test the 'Geocentrism' philosophy from the problem statement.
        
        "지구 자체가 자신의 세상이 되어서 자기가 움직이는 대로 세상이 회전한다."
        "The Earth itself becomes one's world, so the world rotates as one moves."
        """
        manager = FloatingOriginManager(default_sphere_radius=2000.0)
        
        # Universe from elysia's perspective
        manager.register_entity("elysia", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("sun", np.array([1000.0, 0.0, 0.0]))
        manager.register_entity("moon", np.array([-500.0, 0.0, 0.0]))
        
        # Elysia orbits (moves in a circle)
        # Step 1: Move right
        manager.move_entity("elysia", np.array([100.0, 0.0, 0.0]))
        view = manager.get_local_view("elysia")
        
        # From elysia's perspective, the sun moved LEFT (closer)
        assert view["sun"].x < 1000.0
        
        # From elysia's perspective, the moon moved LEFT (further)
        assert view["moon"].x < -500.0

    def test_infinite_universe_precision(self):
        """
        Test that the system can handle infinite universe without precision loss.
        
        "내 좌표는 항상 0이니까, 숫자가 커질 일이 없다."
        "My coordinate is always 0, so numbers never get large."
        """
        manager = FloatingOriginManager(precision_threshold=1e6, default_sphere_radius=1e8)
        manager.register_entity("traveler", np.array([0.0, 0.0, 0.0]))
        manager.register_entity("landmark", np.array([10.0, 0.0, 0.0]))
        
        # Travel a huge distance
        for _ in range(100):
            manager.move_entity("traveler", np.array([1e5, 0.0, 0.0]))
        
        # Relative positions should still be accurate
        view = manager.get_local_view("traveler")
        
        # Traveler is still at origin
        assert view["traveler"].x == 0.0
        
        # Landmark should be far behind
        landmark_pos = view["landmark"]
        assert landmark_pos.x < 0  # Behind the traveler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
