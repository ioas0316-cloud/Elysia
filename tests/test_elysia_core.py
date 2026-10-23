#!/usr/bin/env python3
"""
test_elysia_core.py - Project Elysia Verification Sandbox (Pure Standard Python)
Verifies:
1. Surface Tension - Gravity Equivalence g_eff = (T / rho_Phi) * H
2. Trig-Free Spherical Harmonics (l=0, l=1, l=2) Boundary Evaluation
3. Boundary Crossing & Hysteresis Filtering (anti-chattering)
4. Mock 3-Tier Hierarchical Ring Buffer (SSD -> RAM -> VRAM) & 3GB VRAM Safety
"""

import sys
import math

PI = math.pi
K_Y1 = 0.5 * math.sqrt(3.0 / PI)
K_Y2 = 0.25 * math.sqrt(5.0 / PI)

class MockSpatiotemporalSphere:
    def __init__(self, center=(0.0, 0.0, 0.0), base_radius=10.0, surface_tension=50.0, sh_low=(0.0, 0.0, 0.0, 0.0), flags=2):
        self.center = list(center)
        self.base_radius = float(base_radius)
        self.surface_tension = float(surface_tension)
        self.sh_low = list(sh_low)
        self.flags = int(flags) # bit 0: IsInfiltrated, bit 1: IsActive, bit 2: HasChildren

def vec_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def evaluate_sphere_radius_direction(sphere, n):
    """Trig-free spherical harmonics radius evaluation for direction vector n."""
    norm = vec_len(n)
    if norm > 1e-6:
        nx, ny, nz = n[0]/norm, n[1]/norm, n[2]/norm
    else:
        nx, ny, nz = 0.0, 0.0, 1.0

    R = sphere.base_radius
    # l=1 Dipole
    R += sphere.sh_low[0] * (K_Y1 * ny) + \
         sphere.sh_low[1] * (K_Y1 * nz) + \
         sphere.sh_low[2] * (K_Y1 * nx)
    # l=2 Quadrupole
    R += sphere.sh_low[3] * (K_Y2 * (3.0 * nz * nz - 1.0))
    return max(R, 1e-6)

def compute_surface_tension_gravity(sphere, p_pos, phase_density=1.2, eps=1e-5):
    """Computes g_eff = (T / rho_Phi) * H * direction."""
    r = [p_pos[0] - sphere.center[0], p_pos[1] - sphere.center[1], p_pos[2] - sphere.center[2]]
    dist = vec_len(r)
    if dist < eps:
        return [0.0, 0.0, 0.0]

    n = [r[0]/dist, r[1]/dist, r[2]/dist]
    R_dyn = evaluate_sphere_radius_direction(sphere, n)
    H = 2.0 / R_dyn
    g_mag = (sphere.surface_tension / max(phase_density, eps)) * H
    disp = dist - R_dyn
    atten = math.exp(-abs(disp) * 0.5)

    sign_disp = 1.0 if disp >= 0 else -1.0
    accel_mag = -g_mag * atten * sign_disp
    return [n[0] * accel_mag, n[1] * accel_mag, n[2] * accel_mag]

def process_boundary_crossing(obs_pos, parent_sphere, child_spheres, hysteresis_eps=0.5):
    """Processes infiltration and egress across hyperspherical boundary with hysteresis."""
    rel_pos = [obs_pos[0] - parent_sphere.center[0], obs_pos[1] - parent_sphere.center[1], obs_pos[2] - parent_sphere.center[2]]
    dist = vec_len(rel_pos)
    n = [rel_pos[0]/dist, rel_pos[1]/dist, rel_pos[2]/dist] if dist > 1e-6 else [0.0, 0.0, 1.0]

    dynamic_radius = evaluate_sphere_radius_direction(parent_sphere, n)
    is_currently_inside = bool(parent_sphere.flags & 0x01)

    if not is_currently_inside and (dist <= dynamic_radius - hysteresis_eps):
        parent_sphere.flags |= 0x01 # Infiltrated
        for child in child_spheres:
            child.flags |= 0x02 # Activated
            child.surface_tension += parent_sphere.surface_tension * 0.15
    elif is_currently_inside and (dist > dynamic_radius + hysteresis_eps):
        parent_sphere.flags &= ~0x01 # Egress
        for child in child_spheres:
            child.flags &= ~0x02 # Deactivated

class MockThreeTierMemoryManager:
    """Simulates NVMe SSD (200GB) -> System RAM -> GTX 1060 (3GB VRAM) Tiering."""
    def __init__(self, vram_limit_mb=3072.0, ram_limit_mb=16384.0, ssd_limit_gb=200.0):
        self.vram_limit_bytes = vram_limit_mb * 1024 * 1024
        self.ram_limit_bytes = ram_limit_mb * 1024 * 1024
        self.ssd_limit_bytes = ssd_limit_gb * 1024 * 1024 * 1024

        self.vram_usage_bytes = 0
        self.ram_usage_bytes = 0
        self.ssd_usage_bytes = 0

    def allocate_spheres(self, count=100000):
        # Each sphere is 64 bytes
        bytes_needed = count * 64
        # Allocate in VRAM up to 3GB safety threshold (max 2.5GB for spheres)
        max_vram_for_spheres = 2500 * 1024 * 1024

        vram_count = min(count, int(max_vram_for_spheres // 64))
        remaining = count - vram_count

        ram_count = min(remaining, int((12000 * 1024 * 1024) // 64))
        remaining -= ram_count

        ssd_count = remaining

        self.vram_usage_bytes = vram_count * 64
        self.ram_usage_bytes = ram_count * 64
        self.ssd_usage_bytes = ssd_count * 64

        return vram_count, ram_count, ssd_count

def test_surface_tension_gravity_equivalence():
    print("== 1. Testing Surface Tension Gravity Equivalence ==")
    sphere = MockSpatiotemporalSphere(center=(0, 0, 0), base_radius=10.0, surface_tension=60.0)
    phase_density = 1.5
    # For a spherical monopole R_0 = 10, H = 2 / 10 = 0.2
    # Expected g_mag at boundary surface = (60 / 1.5) * 0.2 = 40 * 0.2 = 8.0

    p_pos = (10.0, 0.0, 0.0)
    g_vec = compute_surface_tension_gravity(sphere, p_pos, phase_density=phase_density)

    expected_g_mag = (60.0 / 1.5) * (2.0 / 10.0)
    calc_g_mag = vec_len(g_vec)

    print(f"  Target boundary acceleration g_mag: {expected_g_mag:.4f}")
    print(f"  Calculated acceleration g_mag:      {calc_g_mag:.4f}")
    assert math.isclose(expected_g_mag, calc_g_mag, rel_tol=1e-4), "Surface tension gravity mismatch!"
    print("  [PASS] Surface Tension Gravity Equivalence Verified.")

def test_trig_free_spherical_harmonics():
    print("== 2. Testing Trig-Free Spherical Harmonics Evaluation ==")
    # Base radius 10.0, Dipole c_1^0 = 2.0 (along z), Quadrupole c_2^0 = 1.0
    sphere = MockSpatiotemporalSphere(center=(0, 0, 0), base_radius=10.0, sh_low=(0.0, 2.0, 0.0, 1.0))

    # Check along Z axis: n = (0, 0, 1)
    # R_z = 10.0 + 2.0 * (K_Y1 * 1.0) + 1.0 * (K_Y2 * (3.0 - 1.0))
    expected_R_z = 10.0 + 2.0 * K_Y1 * 1.0 + 1.0 * K_Y2 * 2.0
    calc_R_z = evaluate_sphere_radius_direction(sphere, [0.0, 0.0, 1.0])

    print(f"  Expected dynamic radius along Z: {expected_R_z:.4f}")
    print(f"  Calculated dynamic radius:      {calc_R_z:.4f}")
    assert math.isclose(expected_R_z, calc_R_z, rel_tol=1e-4), "Spherical harmonic evaluation mismatch!"
    print("  [PASS] Trig-Free Spherical Harmonics Evaluation Verified.")

def test_hysteresis_boundary_crossing():
    print("== 3. Testing Hysteresis Boundary Crossing (Chattering Prevention) ==")
    parent = MockSpatiotemporalSphere(center=(0, 0, 0), base_radius=10.0)
    children = [MockSpatiotemporalSphere(center=(1, 1, 1), base_radius=2.0, flags=0)]

    hysteresis_eps = 0.5
    # Base radius = 10.0. Boundary inside threshold = 9.5, boundary outside threshold = 10.5

    # 1. Approach from outside to dist = 9.8 (between 9.5 and 10.0) -> Should NOT infiltrate yet
    process_boundary_crossing([9.8, 0, 0], parent, children, hysteresis_eps=hysteresis_eps)
    assert not bool(parent.flags & 0x01), "Falsely infiltrated before threshold!"

    # 2. Advance to dist = 9.4 (< 9.5) -> Should INFILTRATE
    process_boundary_crossing([9.4, 0, 0], parent, children, hysteresis_eps=hysteresis_eps)
    assert bool(parent.flags & 0x01), "Failed to infiltrate past threshold!"
    assert bool(children[0].flags & 0x02), "Children failed to activate on infiltration!"

    # 3. Micro-fluctuate back to dist = 9.8 (> 9.5 but < 10.5) -> Should REMAIN infiltrated (Hysteresis prevents chattering)
    process_boundary_crossing([9.8, 0, 0], parent, children, hysteresis_eps=hysteresis_eps)
    assert bool(parent.flags & 0x01), "Chattering detected! Unintentionally egressed!"

    # 4. Move outside to dist = 10.6 (> 10.5) -> Should EGRESS
    process_boundary_crossing([10.6, 0, 0], parent, children, hysteresis_eps=hysteresis_eps)
    assert not bool(parent.flags & 0x01), "Failed to egress past outer threshold!"
    assert not bool(children[0].flags & 0x02), "Children failed to deactivate on egress!"

    print("  [PASS] Hysteresis Boundary Crossing & Anti-Chattering Verified.")

def test_three_tier_memory_mapping():
    print("== 4. Testing 3-Tier Memory Mapping & 3GB VRAM Safety Bound ==")
    mem_mgr = MockThreeTierMemoryManager(vram_limit_mb=3072.0) # GTX 1060

    total_spheres = 50_000_000 # 50 Million Hyperspherical Bubbles (~3.2 GB data)
    vram_cnt, ram_cnt, ssd_cnt = mem_mgr.allocate_spheres(total_spheres)

    vram_mb = (vram_cnt * 64) / (1024 * 1024)
    print(f"  Total Spheres Simulated: {total_spheres:,}")
    print(f"  VRAM Allocated: {vram_cnt:,} spheres ({vram_mb:.2f} MB)")
    print(f"  RAM Allocated:  {ram_cnt:,} spheres ({ram_cnt * 64 / (1024*1024):.2f} MB)")
    print(f"  SSD Deposited:  {ssd_cnt:,} spheres ({ssd_cnt * 64 / (1024*1024*1024):.2f} GB)")

    assert vram_mb <= 2500.0, "VRAM safety limit exceeded!"
    print("  [PASS] 3-Tier Hierarchical Allocation & 3GB VRAM Safety Verified.")

def main():
    print("==================================================")
    print(" Project Elysia Core Verification Sandbox Suite ")
    print("==================================================\n")
    test_surface_tension_gravity_equivalence()
    print()
    test_trig_free_spherical_harmonics()
    print()
    test_hysteresis_boundary_crossing()
    print()
    test_three_tier_memory_mapping()
    print("\n==================================================")
    print(" ALL ELYSIA CORE VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")

if __name__ == "__main__":
    main()
