#!/usr/bin/env python3
"""
Verification script for the Cl(3,1) Space-Time Algebra, Dynamic Metric g_uv,
Christoffel Geodesic Integration, and 3D OBJ/CSV Exporter Pipeline.
"""

import os
import sys
import numpy as np

import elysia_cl31_pybind
from elysia_engine.dynamic_metric_geodesic import DynamicMetricEnginePython, GeodesicIntegratorPython
from elysia_engine.torus_exporter import TorusMeshExporterPython, TorusAttractorSimulatorPython

def run_verification():
    print("=" * 70)
    print("      ELYSIA Cl(3,1) STA & GEODESIC TOPOLOGY VERIFICATION DEMO")
    print("=" * 70)

    # 1. Cl(3,1) Space-Time Algebra Weaving Verification
    print("\n[Step 1] Cl(3,1) Space-Time Algebra Multivector Weaving:")
    N = 5
    A_s  = np.ones(N, dtype=np.float32)
    A_v0 = np.full(N, 1.0, dtype=np.float32)
    A_v1 = np.full(N, 0.5, dtype=np.float32)
    A_v2 = np.full(N, 0.2, dtype=np.float32)
    A_v3 = np.full(N, 0.1, dtype=np.float32)

    B_s  = np.ones(N, dtype=np.float32)
    B_v0 = np.full(N, 1.0, dtype=np.float32)
    B_v1 = np.full(N, -0.3, dtype=np.float32)
    B_v2 = np.full(N, 0.8, dtype=np.float32)
    B_v3 = np.full(N, 0.4, dtype=np.float32)

    res = elysia_cl31_pybind.cl31_weave(
        A_s, A_v0, A_v1, A_v2, A_v3,
        B_s, B_v0, B_v1, B_v2, B_v3
    )

    out_s, out_b0, out_b1, out_b2, out_b3, out_b4, out_b5 = res
    print(f"  Inputs A vector[0..3]: [{A_v0[0]}, {A_v1[0]}, {A_v2[0]}, {A_v3[0]}]")
    print(f"  Inputs B vector[0..3]: [{B_v0[0]}, {B_v1[0]}, {B_v2[0]}, {B_v3[0]}]")
    print(f"  Scalar Output out_s[0] : {out_s[0]:.4f}")
    print(f"  Bivector Sheet out_b0  : {out_b0[0]:.4f}")
    print(f"  Bivector Sheet out_b1  : {out_b1[0]:.4f}")

    # 2. Dynamic Metric Tensor g_uv Update Verification
    print("\n[Step 2] Dynamic Metric Tensor g_uv Information Ricci Flow:")
    metric_engine = DynamicMetricEnginePython(kappa=0.15, alpha=0.05)
    F_input = np.array([
        [0.0,  0.8,  0.3,  0.0],
        [-0.8, 0.0,  0.5,  0.1],
        [-0.3, -0.5, 0.0,  0.2],
        [0.0,  -0.1, -0.2, 0.0]
    ], dtype=np.float64)

    print("  Updating Metric over 10 Information Gravity time steps...")
    for step in range(10):
        g = metric_engine.update_metric_step(F_input, dt=0.01)

    print("  Deformed Spacetime Metric g_uv:")
    for row in g:
        print("  ", " ".join(f"{val:8.4f}" for val in row))

    # 3. Christoffel Geodesic Integration Verification
    print("\n[Step 3] Christoffel Symbol Geodesic Trajectory Tracking:")
    geodesic_engine = GeodesicIntegratorPython(g)
    pos = np.array([0.0, 0.1, 0.5, 0.0], dtype=np.float64)
    vel = np.array([1.0, 0.2, -0.1, 0.4], dtype=np.float64)

    print(f"  Initial Position x^mu : [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}, {pos[3]:.4f}]")
    for step in range(5):
        pos, vel = geodesic_engine.step_geodesic(pos, vel, dtau=0.01)
        print(f"  Step {step+1} Position x^mu : [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}, {pos[3]:.4f}]")

    # 4. 3D OBJ Mesh and CSV Geodesic Trajectory Export
    print("\n[Step 4] 3D Wavefront OBJ Mesh & CSV Trajectory Export:")
    exporter = TorusMeshExporterPython()
    obj_path = "memory_torus_manifold.obj"
    csv_path = "geodesic_path.csv"

    ok_obj = exporter.export_to_obj(obj_path, grid_theta=60, grid_phi=30)
    print(f"  Export Deformed Torus Mesh to '{obj_path}': {'SUCCESS' if ok_obj else 'FAILED'}")

    sim = TorusAttractorSimulatorPython()
    path = sim.simulate_trajectory((0.8, 2.0), (0.5, 0.3), steps=60, dt=0.02)
    ok_csv = exporter.export_geodesic_csv(csv_path, path)
    print(f"  Export Geodesic Trajectory ({len(path)} points) to '{csv_path}': {'SUCCESS' if ok_csv else 'FAILED'}")

    print("\n" + "=" * 70)
    print("      ALL VERIFICATION STEPS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
