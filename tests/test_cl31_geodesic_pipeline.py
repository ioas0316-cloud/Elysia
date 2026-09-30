import os
import unittest
import numpy as np

# Import custom modules
import elysia_cl31_pybind
from elysia_engine.dynamic_metric_geodesic import DynamicMetricEnginePython, GeodesicIntegratorPython
from elysia_engine.torus_exporter import TorusMeshExporterPython, TorusAttractorSimulatorPython

class TestCl31STAEngine(unittest.TestCase):
    def test_cl31_weave_algebra(self):
        A_s  = np.array([1.0], dtype=np.float32)
        A_v0 = np.array([1.0], dtype=np.float32)
        A_v1 = np.array([0.5], dtype=np.float32)
        A_v2 = np.array([0.0], dtype=np.float32)
        A_v3 = np.array([0.0], dtype=np.float32)

        B_s  = np.array([1.0], dtype=np.float32)
        B_v0 = np.array([1.0], dtype=np.float32)
        B_v1 = np.array([-0.3], dtype=np.float32)
        B_v2 = np.array([0.0], dtype=np.float32)
        B_v3 = np.array([0.0], dtype=np.float32)

        res = elysia_cl31_pybind.cl31_weave(
            A_s, A_v0, A_v1, A_v2, A_v3,
            B_s, B_v0, B_v1, B_v2, B_v3
        )

        out_s, out_b0, out_b1, out_b2, out_b3, out_b4, out_b5 = res
        # out_s = A_s*B_s + A_v0*B_v0 - A_v1*B_v1 = 1.0*1.0 + 1.0*1.0 - 0.5*(-0.3) = 2.15
        self.assertAlmostEqual(out_s[0], 2.15, places=5)
        # out_b0 (e01) = A_v0*B_v1 - A_v1*B_v0 = 1.0*(-0.3) - 0.5*1.0 = -0.8
        self.assertAlmostEqual(out_b0[0], -0.8, places=5)

class TestDynamicMetricGeodesic(unittest.TestCase):
    def test_dynamic_metric_update(self):
        engine = DynamicMetricEnginePython(kappa=0.15, alpha=0.05)
        F_bivector = np.array([
            [0.0,  0.8,  0.3,  0.0],
            [-0.8, 0.0,  0.5,  0.1],
            [-0.3, -0.5, 0.0,  0.2],
            [0.0,  -0.1, -0.2, 0.0]
        ], dtype=np.float64)

        updated_g = engine.update_metric_step(F_bivector, dt=0.01)
        self.assertEqual(updated_g.shape, (4, 4))
        # Metric should deform from flat Minkowski [1.0, -1.0, -1.0, -1.0]
        self.assertNotEqual(updated_g[0, 0], 1.0)

    def test_geodesic_trajectory(self):
        g = np.diag([0.95, -1.08, -1.02, -0.91]).astype(np.float64)
        integrator = GeodesicIntegratorPython(g)
        pos = np.array([0.0, 0.1, 0.5, 0.0], dtype=np.float64)
        vel = np.array([1.0, 0.2, -0.1, 0.4], dtype=np.float64)

        new_pos, new_vel = integrator.step_geodesic(pos.copy(), vel.copy(), dtau=0.01)
        self.assertFalse(np.array_equal(pos, new_pos))

class TestTorus3DExporter(unittest.TestCase):
    def test_obj_and_csv_export(self):
        exporter = TorusMeshExporterPython()
        obj_file = "test_memory_torus.obj"
        csv_file = "test_geodesic_path.csv"

        ok_obj = exporter.export_to_obj(obj_file, grid_theta=20, grid_phi=10)
        self.assertTrue(ok_obj)
        self.assertTrue(os.path.exists(obj_file))

        sim = TorusAttractorSimulatorPython()
        path = sim.simulate_trajectory((0.8, 2.0), (0.5, 0.3), steps=20, dt=0.02)
        self.assertEqual(len(path), 20)

        ok_csv = exporter.export_geodesic_csv(csv_file, path)
        self.assertTrue(ok_csv)
        self.assertTrue(os.path.exists(csv_file))

        # Cleanup
        if os.path.exists(obj_file): os.remove(obj_file)
        if os.path.exists(csv_file): os.remove(csv_file)

if __name__ == '__main__':
    unittest.main()
