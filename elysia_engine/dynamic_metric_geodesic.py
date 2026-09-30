import numpy as np

class DynamicMetricEnginePython:
    """
    Python implementation of Dynamic Spacetime Metric g_uv with Information Ricci Flow
    and Information Energy-Momentum Tensor T_uv.
    """
    def __init__(self, kappa=0.15, alpha=0.05):
        self.kappa = kappa
        self.alpha = alpha
        # Minkowski flat metric eta_uv = diag(1, -1, -1, -1)
        self.g = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

    def compute_stress_energy_tensor(self, F_bivector: np.ndarray) -> np.ndarray:
        """
        Compute T_uv^(info) from bivector sheet F_uv (4x4 matrix).
        T_uv = F_ua * F_v^a - 1/4 * g_uv * (F_ab * F^ab)
        """
        g_diag = np.diag(self.g)
        # F_sq = F_ab * F^ab
        F_sq = 0.0
        for a in range(4):
            for b in range(4):
                F_sq += F_bivector[a, b] * F_bivector[a, b] * g_diag[a] * g_diag[b]

        T = np.zeros((4, 4), dtype=np.float64)
        for u in range(4):
            for v in range(4):
                interaction = 0.0
                for a in range(4):
                    interaction += F_bivector[u, a] * F_bivector[v, a] * g_diag[a]
                T[u, v] = interaction - 0.25 * self.g[u, v] * F_sq
        return T

    def update_metric_step(self, F_bivector: np.ndarray, dt: float):
        """
        g_uv(t + dt) = g_uv(t) + dt * (-2 alpha * R_uv + kappa * T_uv)
        """
        T_info = self.compute_stress_energy_tensor(F_bivector)
        target_g = np.diag([1.0, -1.0, -1.0, -1.0])
        R_uv = self.g - target_g  # Proportional Ricci curvature deviation

        dg_dt = -2.0 * self.alpha * R_uv + self.kappa * T_info
        self.g += dt * dg_dt
        return self.g


class GeodesicIntegratorPython:
    """
    Python implementation of Christoffel Symbols Gamma^l_mn and Geodesic Trajectory Integration.
    """
    def __init__(self, metric: np.ndarray):
        self.g = metric
        self.g_inv = np.linalg.inv(metric)
        self.Gamma = np.zeros((4, 4, 4), dtype=np.float64)
        self._compute_christoffel_symbols()

    def _compute_christoffel_symbols(self):
        # Spatial curvature gradient modeling memory attractor valley
        dg = np.zeros((4, 4, 4), dtype=np.float64)
        dg[1, 1, 1] = 0.05
        dg[2, 2, 2] = -0.03
        dg[3, 3, 3] = 0.08
        dg[1, 0, 1] = 0.02
        dg[0, 1, 1] = 0.02

        for l in range(4):
            for m in range(4):
                for n in range(4):
                    sum_val = 0.0
                    for s in range(4):
                        term = dg[m, s, n] + dg[n, s, m] - dg[s, m, n]
                        sum_val += 0.5 * self.g_inv[l, s] * term
                    self.Gamma[l, m, n] = sum_val

    def step_geodesic(self, position: np.ndarray, velocity: np.ndarray, dtau: float):
        """
        dv^l / dtau = - Gamma^l_mn * v^m * v^n
        dx^i / dtau = v^i
        """
        acceleration = np.zeros(4, dtype=np.float64)
        for l in range(4):
            for m in range(4):
                for n in range(4):
                    acceleration[l] -= self.Gamma[l, m, n] * velocity[m] * velocity[n]

        position += velocity * dtau
        velocity += acceleration * dtau
        return position, velocity
