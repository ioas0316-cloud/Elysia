import numpy as np

class TopologicalManifold:
    """
    A continuous spatiotemporal surface (manifold) representing the causal field.
    Instead of calculating variances of discrete points, this simulates a 2D mesh
    where information creates waves. Resonance is determined by constructive/destructive
    interference resulting in equilibrium (low surface tension).
    """
    def __init__(self, size=64, damping=0.98, wave_speed=0.4):
        self.size = size
        self.damping = damping
        self.wave_speed = wave_speed
        
        # u is the current height (tension/displacement) of the surface
        self.u = np.zeros((size, size))
        self.u_prev = np.zeros((size, size))
        
    def inject_disturbance(self, x_normalized: float, y_normalized: float, amplitude: float = 1.0, radius: int = 4):
        """
        Inject a wave (like a drop of water or a chunk of information) onto the surface.
        x_normalized, y_normalized are floats between 0.0 and 1.0.
        """
        cx = int(x_normalized * (self.size - 1))
        cy = int(y_normalized * (self.size - 1))
        
        # Apply a gaussian pulse
        for i in range(max(0, cx - radius), min(self.size, cx + radius + 1)):
            for j in range(max(0, cy - radius), min(self.size, cy + radius + 1)):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist <= radius:
                    self.u[i, j] += amplitude * (1 - dist/radius)
                    
    def step(self):
        """
        Advance the physics simulation by one time step using a discrete Laplace operator.
        """
        # Laplace operator using finite difference
        laplacian = (
            np.roll(self.u, 1, axis=0) + np.roll(self.u, -1, axis=0) +
            np.roll(self.u, 1, axis=1) + np.roll(self.u, -1, axis=1) - 4 * self.u
        )
        
        # Fixed boundaries
        laplacian[0, :] = laplacian[-1, :] = laplacian[:, 0] = laplacian[:, -1] = 0
        
        # Wave equation: u_next = 2*u - u_prev + (c^2) * laplacian
        u_next = 2 * self.u - self.u_prev + (self.wave_speed ** 2) * laplacian
        
        # Apply damping (entropy/friction)
        u_next *= self.damping
        
        # Shift time
        self.u_prev = self.u.copy()
        self.u = u_next
        
    def calculate_surface_tension(self) -> float:
        """
        Total surface tension across the manifold (gradient magnitude).
        A low tension implies structural equilibrium (Resonance).
        A high tension implies chaotic friction and dissonance.
        """
        grad_x = np.diff(self.u, axis=0)
        grad_y = np.diff(self.u, axis=1)
        
        tension = np.sum(grad_x**2) + np.sum(grad_y**2)
        return float(tension)

    def get_state(self):
        """Returns the current manifold state for visualization"""
        return self.u.tolist()
