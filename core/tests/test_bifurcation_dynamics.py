"""
[VERIFICATION: TENSION-DRIVEN DIMENSIONAL SPLIT]
Verifies dynamic bifurcation (axes split) and compression in fractal rotors.
"""

import math
import sys
import os

# Ensure import paths are resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fractal_rotor import Rotor

def test_bifurcation_on_high_tension():
    """Verify that a rotor splits its dimensions when tension exceeds limits."""
    # Create parent and child rotor
    parent = Rotor("P", level=0)
    child = Rotor("C", level=1, parent=parent)
    parent.attach_child(child)

    # Initial state
    assert child.active_axes == 3
    assert child.stable_ticks == 0

    # Inject high tension (tension_limit for level 1 is (pi/2) / 2 = pi/4 ~ 0.785)
    # Let's set child phase offset to 1.2 radians, which exceeds the limit
    child.phase_offset = 1.2
    
    # Observe phase alignment. High tension triggers bifurcation.
    parent.observe()

    # The active axes should expand from 3 to 4
    assert child.active_axes == 4
    # The phase offset should have been reduced/distributed by bifurcate
    assert child.phase_offset < 1.2
    # stable ticks should remain reset to 0
    assert child.stable_ticks == 0


def test_compression_on_stability():
    """Verify that a rotor compresses dimensions back down on stable phase alignment."""
    parent = Rotor("P", level=0)
    child = Rotor("C", level=1, parent=parent)
    parent.attach_child(child)

    # Manually set axes to 4 to test reduction
    child.active_axes = 4

    # Keep phase offset very small (stable condition)
    # tension_limit is pi/4 ~ 0.785. 20% of tension_limit is ~0.157.
    # We set offset to 0.05
    child.phase_offset = 0.05
    child.stable_ticks = 0

    # Run for 5 steps to trigger compression
    for i in range(5):
        parent.observe()
        # Verify stable ticks accumulate
        assert child.stable_ticks == i + 1 or child.active_axes == 3

    # On the 5th tick, active axes should compress from 4 to 3
    assert child.active_axes == 3
    # stable ticks should reset to 0 upon compression
    assert child.stable_ticks == 0


def test_collapse_fallback_at_limit():
    """Verify that normal realign collapse occurs if active axes hits MAX_AXES (8)."""
    parent = Rotor("P", level=0)
    child = Rotor("C", level=1, parent=parent)
    parent.attach_child(child)

    # Force active axes to MAX_AXES
    child.active_axes = 8
    assert child.active_axes == child.MAX_AXES

    # Inject extreme tension (e.g. 1.5 radians)
    child.phase_offset = 1.5

    # Since axes is at max limit, observe must trigger collapse_and_realign instead of bifurcating
    parent.observe()

    # active axes must remain at 8
    assert child.active_axes == 8
    # It must have collapsed and realigned to a stable attractor (0.0 or pi)
    # Since 1.5 is closer to pi (3.14) or 0?
    # Wait, in collapse_and_realign: if abs(offset) > pi/2 (1.57), it goes to pi.
    # 1.5 is slightly below pi/2, so it collapses to 0.0.
    assert child.phase_offset == 0.0
    assert child.tension == 0.0
