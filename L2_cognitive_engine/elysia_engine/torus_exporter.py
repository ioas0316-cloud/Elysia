import math
import numpy as np
from typing import List, Tuple

class TorusMeshExporterPython:
    """
    Python exporter for 3D OBJ deformed Torus manifold mesh and CSV geodesic particle trajectory logs.
    """
    def __init__(self, R: float = 3.0, r: float = 1.0, k_strength: float = 0.8, alpha: float = 1.5):
        self.R = R
        self.r = r
        self.theta_0 = math.pi / 2.0  # 1.5708
        self.phi_0 = math.pi         # 3.1415
        self.k_strength = k_strength
        self.alpha = alpha

    def compute_deformed_torus_point(self, theta: float, phi: float) -> Tuple[float, float, float]:
        dist_sq = (theta - self.theta_0) ** 2 + (phi - self.phi_0) ** 2
        local_r = self.r - self.k_strength * math.exp(-self.alpha * dist_sq)

        x = (self.R + local_r * math.cos(phi)) * math.cos(theta)
        y = (self.R + local_r * math.cos(phi)) * math.sin(theta)
        z = local_r * math.sin(phi)
        return (x, y, z)

    def export_to_obj(self, filename: str, grid_theta: int = 60, grid_phi: int = 30) -> bool:
        try:
            with open(filename, 'w') as f:
                d_theta = 2.0 * math.pi / grid_theta
                d_phi = 2.0 * math.pi / grid_phi

                # Vertices
                for i in range(grid_theta):
                    theta = i * d_theta
                    for j in range(grid_phi):
                        phi = j * d_phi
                        x, y, z = self.compute_deformed_torus_point(theta, phi)
                        f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

                # Quad Faces
                for i in range(grid_theta):
                    for j in range(grid_phi):
                        next_i = (i + 1) % grid_theta
                        next_j = (j + 1) % grid_phi

                        idx1 = i * grid_phi + j + 1
                        idx2 = next_i * grid_phi + j + 1
                        idx3 = next_i * grid_phi + next_j + 1
                        idx4 = i * grid_phi + next_j + 1

                        f.write(f"f {idx1} {idx2} {idx3} {idx4}\n")
            return True
        except Exception as e:
            print(f"Error exporting OBJ: {e}")
            return False

    @staticmethod
    def export_geodesic_csv(filename: str, trajectory: List[Tuple[float, float, float]]) -> bool:
        try:
            with open(filename, 'w') as f:
                f.write("step,x,y,z\n")
                for step, (x, y, z) in enumerate(trajectory):
                    f.write(f"{step},{x:.6f},{y:.6f},{z:.6f}\n")
            return True
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False


class TorusAttractorSimulatorPython:
    def __init__(self, R: float = 3.0, r: float = 1.0, k_strength: float = 2.5, alpha: float = 1.5):
        self.R = R
        self.r = r
        self.theta_0 = math.pi / 2.0
        self.phi_0 = math.pi
        self.k_strength = k_strength
        self.alpha = alpha
        self.exporter = TorusMeshExporterPython(R, r, 0.8)

    def g_tt(self, theta: float, phi: float) -> float:
        dist_sq = (theta - self.theta_0) ** 2 + (phi - self.phi_0) ** 2
        return (self.R + self.r * math.cos(phi)) ** 2 + self.k_strength * math.exp(-self.alpha * dist_sq)

    def g_pp(self, theta: float, phi: float) -> float:
        dist_sq = (theta - self.theta_0) ** 2 + (phi - self.phi_0) ** 2
        return self.r ** 2 + self.k_strength * math.exp(-self.alpha * dist_sq)

    def compute_christoffel(self, theta: float, phi: float) -> np.ndarray:
        eps = 1e-5
        dg_tt_dtheta = (self.g_tt(theta + eps, phi) - self.g_tt(theta - eps, phi)) / (2.0 * eps)
        dg_tt_dphi   = (self.g_tt(theta, phi + eps) - self.g_tt(theta, phi - eps)) / (2.0 * eps)
        dg_pp_dtheta = (self.g_pp(theta + eps, phi) - self.g_pp(theta - eps, phi)) / (2.0 * eps)
        dg_pp_dphi   = (self.g_pp(theta, phi + eps) - self.g_pp(theta, phi - eps)) / (2.0 * eps)

        g_tt_inv = 1.0 / self.g_tt(theta, phi)
        g_pp_inv = 1.0 / self.g_pp(theta, phi)

        Gamma = np.zeros((2, 2, 2), dtype=np.float64)
        Gamma[0, 0, 0] = 0.5 * g_tt_inv * dg_tt_dtheta
        Gamma[0, 0, 1] = 0.5 * g_tt_inv * dg_tt_dphi
        Gamma[0, 1, 0] = Gamma[0, 0, 1]
        Gamma[0, 1, 1] = -0.5 * g_tt_inv * dg_pp_dtheta

        Gamma[1, 0, 0] = -0.5 * g_pp_inv * dg_tt_dphi
        Gamma[1, 0, 1] = 0.5 * g_pp_inv * dg_pp_dtheta
        Gamma[1, 1, 0] = Gamma[1, 0, 1]
        Gamma[1, 1, 1] = 0.5 * g_pp_inv * dg_pp_dphi

        return Gamma

    def simulate_trajectory(self, start_pos: Tuple[float, float], start_vel: Tuple[float, float], steps: int, dt: float) -> List[Tuple[float, float, float]]:
        theta, phi = start_pos
        dtheta, dphi = start_vel
        trajectory = []

        for _ in range(steps):
            Gamma = self.compute_christoffel(theta, phi)

            acc_theta = -(Gamma[0, 0, 0] * dtheta * dtheta +
                          2.0 * Gamma[0, 0, 1] * dtheta * dphi +
                          Gamma[0, 1, 1] * dphi * dphi)

            acc_phi   = -(Gamma[1, 0, 0] * dtheta * dtheta +
                          2.0 * Gamma[1, 0, 1] * dtheta * dphi +
                          Gamma[1, 1, 1] * dphi * dphi)

            theta += dtheta * dt
            phi   += dphi * dt
            dtheta += acc_theta * dt
            dphi   += acc_phi * dt

            pt3d = self.exporter.compute_deformed_torus_point(theta, phi)
            trajectory.append(pt3d)

        return trajectory
