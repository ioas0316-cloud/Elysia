"""
Unit tests for 2D Spacetime Field & Liquid Blistering Simulation.

Verifies:
1. Sub-critical separation keeps blobs structurally independent without snap.
2. Super-critical in-phase resonance produces liquid blister snapping and bridge formation.
3. Anti-phase interference produces deconstructive repulsion preventing merger.
4. Edge sharpness (gradient magnitude) is significantly higher than naive Gaussian blur.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_step2_2d_field_blister_simulation import (
    compute_blob_sdf,
    compute_blister_field_2d,
)


def test_2d_field_subcritical_separation():
    """Verify that far-apart blobs remain separated without blister snap."""
    x = np.linspace(-1.5, 1.5, 80)
    y = np.linspace(-1.0, 1.0, 80)
    grid_x, grid_y = np.meshgrid(x, y)

    res = compute_blister_field_2d(
        grid_x, grid_y,
        center_a=(-0.7, 0.0), center_b=(0.7, 0.0),
        radius_a=0.45, radius_b=0.45,
        phase_a=0.0, phase_b=0.0,
    )

    assert res["overlap"] < 0.25, "Overlap should be below critical threshold"
    assert res["snap_intensity"] == 0.0, "Snap intensity must be 0 in sub-critical regime"

    # Midpoint (0, 0) should be outside both blobs (intensity close to 0)
    mid_idx_x = len(x) // 2
    mid_idx_y = len(y) // 2
    mid_intensity = res["intensity_blister"][mid_idx_y, mid_idx_x]
    assert mid_intensity < 0.05, f"Midpoint intensity should be near zero, got {mid_intensity}"


def test_2d_field_supercritical_inphase_snap():
    """Verify that in-phase blobs past critical overlap snap together into a unified liquid bridge."""
    x = np.linspace(-1.5, 1.5, 80)
    y = np.linspace(-1.0, 1.0, 80)
    grid_x, grid_y = np.meshgrid(x, y)

    res = compute_blister_field_2d(
        grid_x, grid_y,
        center_a=(-0.3, 0.0), center_b=(0.3, 0.0),
        radius_a=0.45, radius_b=0.45,
        phase_a=0.0, phase_b=0.0,
    )

    assert res["overlap"] > 0.25, "Overlap must exceed critical threshold Omega_c"
    assert res["cos_interf"] == pytest.approx(1.0), "In-phase cosine factor must be 1.0"
    assert res["snap_intensity"] > 0.0, "Snap intensity must be strictly positive"

    # Midpoint (0, 0) should now have high optical intensity due to the snapped bridge
    mid_idx_x = len(x) // 2
    mid_idx_y = len(y) // 2
    mid_intensity = res["intensity_blister"][mid_idx_y, mid_idx_x]
    assert mid_intensity > 0.90, f"Midpoint should be solidified by blister bridge, got {mid_intensity}"


def test_2d_field_antiphase_deconstruction():
    """Verify that anti-phase blobs resist merging and exhibit deconstructive separation."""
    x = np.linspace(-1.5, 1.5, 80)
    y = np.linspace(-1.0, 1.0, 80)
    grid_x, grid_y = np.meshgrid(x, y)

    res_inphase = compute_blister_field_2d(
        grid_x, grid_y,
        center_a=(-0.35, 0.0), center_b=(0.35, 0.0),
        radius_a=0.45, radius_b=0.45,
        phase_a=0.0, phase_b=0.0,
    )

    res_antiphase = compute_blister_field_2d(
        grid_x, grid_y,
        center_a=(-0.35, 0.0), center_b=(0.35, 0.0),
        radius_a=0.45, radius_b=0.45,
        phase_a=0.0, phase_b=np.pi,
    )

    assert res_antiphase["cos_interf"] == pytest.approx(-1.0), "Anti-phase factor must be -1.0"
    assert res_antiphase["snap_intensity"] < 0.0, "Snap intensity must be negative (repulsive barrier)"

    # Midpoint intensity under anti-phase should be significantly lower than in-phase
    mid_idx_x = len(x) // 2
    mid_idx_y = len(y) // 2
    assert res_antiphase["intensity_blister"][mid_idx_y, mid_idx_x] < res_inphase["intensity_blister"][mid_idx_y, mid_idx_x]


def test_2d_field_edge_sharpness_vs_blur():
    """Verify that blister kernel preserves higher gradient magnitude (edge sharpness) than blur."""
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.0, 1.0, 100)
    grid_x, grid_y = np.meshgrid(x, y)

    res = compute_blister_field_2d(
        grid_x, grid_y,
        center_a=(-0.325, 0.0), center_b=(0.325, 0.0),
        radius_a=0.45, radius_b=0.45,
        phase_a=0.0, phase_b=0.0,
    )

    max_grad_blister = float(np.max(res["grad_mag_blister"]))
    max_grad_blur = float(np.max(res["grad_mag_blur"]))

    assert max_grad_blister > 1.5 * max_grad_blur, (
        f"Blister gradient ({max_grad_blister:.4f}) should be significantly sharper "
        f"than blur gradient ({max_grad_blur:.4f})"
    )
