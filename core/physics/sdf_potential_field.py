"""
SDF & Potential Field Coupling Engine
=====================================
Philosophy: "Do not calculate, let it flow."
Eliminates branching (if-else) by encoding physical boundaries, collision barriers,
attractors, and light/shadow fields into a unified, continuous, differentiable scalar field U(x).

Key Features:
1. Polynomial Smooth Minimum (smin) for organic blob-like object blending (metaball effect).
2. Prime SDF primitives (Sphere/Circle, Box/Rectangle, Infinite Plane) for 2D & 3D space.
3. Continuous Huber Attractor Potential to prevent long-distance velocity explosion.
4. C1 Continuous Phototaxis (Light seeking / Shadow avoidance) via Sphere Tracing Ray Marching.
5. Continuous Inward-masked Sliding Projection to slide along boundary surfaces without sticking or if-else blocks.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# -------------------------------------------------------------
# 1. Core Mathematical Utilities
# -------------------------------------------------------------

def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """
    Inigo Quilez's Polynomial Smooth Minimum.
    Blends two distance fields smoothly using a parameter k.
    k = 0 degenerates to standard min(a, b).
    """
    if k <= 1e-9:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """
    Standard smoothstep function for continuous C1 interpolation.
    """
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# -------------------------------------------------------------
# 2. SDF Primitives & Scene Composition
# -------------------------------------------------------------

class SDFPrimitive:
    """Base class for analytical Signed Distance Fields."""
    def evaluate(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SDFSphere(SDFPrimitive):
    """SDF for a Sphere in 3D or a Circle in 2D."""
    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        # p: (N, D) where D is 2 or 3
        diff = p - self.center
        dist = np.linalg.norm(diff, axis=-1)
        return dist - self.radius

class SDFBox(SDFPrimitive):
    """SDF for an Axis-Aligned Box (Box in 3D / Rectangle in 2D)."""
    def __init__(self, center: np.ndarray, half_extents: np.ndarray):
        self.center = np.array(center, dtype=np.float32)
        self.half_extents = np.array(half_extents, dtype=np.float32)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        # p: (N, D)
        d = np.abs(p - self.center) - self.half_extents
        outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        inside_dist = np.minimum(np.max(d, axis=-1), 0.0)
        return outside_dist + inside_dist

class SDFPlane(SDFPrimitive):
    """SDF for an infinite Plane defined by a point on plane and normal vector."""
    def __init__(self, point: np.ndarray, normal: np.ndarray):
        self.point = np.array(point, dtype=np.float32)
        # Normalize the plane normal
        n = np.array(normal, dtype=np.float32)
        self.normal = n / (np.linalg.norm(n) + 1e-9)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        # p: (N, D)
        return np.sum((p - self.point) * self.normal, axis=-1)


class SDFScene:
    """
    Composite scene of multiple SDF primitives blended together via smin.
    Provides analytical and finite-difference gradients.
    """
    def __init__(self, blend_k: float = 0.5):
        self.primitives: List[SDFPrimitive] = []
        self.blend_k = blend_k

    def add(self, primitive: SDFPrimitive):
        self.primitives.append(primitive)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """Evaluates the blended SDF for all points in p (shape: (N, D))."""
        if not self.primitives:
            return np.ones(p.shape[0], dtype=np.float32) * 1e5

        d = self.primitives[0].evaluate(p)
        for prim in self.primitives[1:]:
            d_next = prim.evaluate(p)
            d = smin(d, d_next, self.blend_k)
        return d

    def evaluate_gradient(self, p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Computes the spatial gradient (Normal) of the composite SDF scene
        using central finite differences to ensure continuous, correct normals.
        Returns array of shape (N, D) with normalized gradients.
        """
        N, D = p.shape
        grad = np.zeros_like(p)

        # Central difference along each dimension
        for i in range(D):
            eps_vec = np.zeros(D, dtype=np.float32)
            eps_vec[i] = eps

            d_plus = self.evaluate(p + eps_vec)
            d_minus = self.evaluate(p - eps_vec)
            grad[:, i] = (d_plus - d_minus) / (2.0 * eps)

        norms = np.linalg.norm(grad, axis=-1, keepdims=True)
        # Avoid division by zero, fallback to normal vector pointing out or 0
        safe_norms = np.where(norms > 1e-6, norms, 1.0)
        grad_normalized = grad / safe_norms
        return grad_normalized


# -------------------------------------------------------------
# 3. Dynamic Field Kinematics & Energy Master Field
# -------------------------------------------------------------

class SDFPotentialFieldSimulation:
    """
    Fully vectorized simulation of particle systems moving under a Unified Master Field:
    U(x) = w_attractor * V_attractor(x) + w_collision * V_collision(x) + w_light * V_light(x)
    """
    def __init__(
        self,
        scene: SDFScene,
        dimensions: int = 3,
        w_attractor: float = 1.0,
        w_collision: float = 2.0,
        w_light: float = 0.5,
        k_repulsive: float = 100.0,
        k_shadow: float = 4.0,
        epsilon_huber: float = 0.1,
        collision_threshold: float = 0.1
    ):
        self.scene = scene
        self.dims = dimensions
        self.w_attractor = w_attractor
        self.w_collision = w_collision
        self.w_light = w_light
        self.k_repulsive = k_repulsive
        self.k_shadow = k_shadow
        self.epsilon_huber = epsilon_huber
        self.collision_threshold = collision_threshold # Distance threshold to trigger sliding behavior

    def evaluate_V_attractor(self, p: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Huber-like potential to avoid stiffness and speed explosion at huge distances.
        V = sqrt(||x - x_target||^2 + eps^2) - eps
        """
        diff = p - target
        dist_sq = np.sum(diff**2, axis=-1)
        return np.sqrt(dist_sq + self.epsilon_huber**2) - self.epsilon_huber

    def evaluate_grad_V_attractor(self, p: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Gradient of V_attractor: diff / sqrt(||diff||^2 + eps^2)."""
        diff = p - target
        dist_sq = np.sum(diff**2, axis=-1, keepdims=True)
        return diff / np.sqrt(dist_sq + self.epsilon_huber**2)

    def evaluate_V_collision(self, phi: np.ndarray) -> np.ndarray:
        """
        Elastic potential for collision barriers: 0.5 * k * [min(0, phi)]^2.
        No if-else branching.
        """
        overlap = np.minimum(0.0, phi)
        return 0.5 * self.k_repulsive * (overlap ** 2)

    def evaluate_grad_V_collision(self, phi: np.ndarray, grad_phi: np.ndarray) -> np.ndarray:
        """
        Gradient of collision potential: k * min(0, phi) * grad_phi.
        """
        overlap = np.minimum(0.0, phi)
        return self.k_repulsive * overlap[:, np.newaxis] * grad_phi

    def compute_soft_shadow(
        self,
        p: np.ndarray,
        light_dir: np.ndarray,
        max_steps: int = 16,
        min_t: float = 0.02,
        max_t: float = 5.0
    ) -> np.ndarray:
        """
        Sphere-tracing soft shadow coefficient tracking:
        S(x) = min_{t > 0} ( k_shadow * phi(x + t * d) / t )
        Optimized and fully vectorized using Ray Marching across batch dimension.
        """
        N = p.shape[0]
        # Current ray distance
        t = np.ones(N, dtype=np.float32) * min_t
        shadow = np.ones(N, dtype=np.float32)

        # Ray marching loop
        for _ in range(max_steps):
            # Compute current ray position: p_curr = p + t * light_dir
            p_curr = p + t[:, np.newaxis] * light_dir[np.newaxis, :]

            # Distance from scene
            dist = self.scene.evaluate(p_curr)

            # Avoid division by zero near the start
            h = np.maximum(dist, 0.0)
            step_shadow = (self.k_shadow * h) / (t + 1e-8)
            shadow = np.minimum(shadow, step_shadow)

            # Step forward by distance (Sphere Tracing benefit)
            t += np.maximum(dist, 0.01)

            # Terminate rays that exceed max distance
            t = np.minimum(t, max_t)

        return np.clip(shadow, 0.0, 1.0)

    def evaluate_V_light(self, p: np.ndarray, light_dir: np.ndarray, grad_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        C1 Continuous Phototaxis (light-seeking, shadow avoidance).
        Using Smoothplus for diffuse shading:
        I_diffuse = 0.5 * ( (N . L) + sqrt((N . L)^2 + eps^2) )
        V_light = - S * I_diffuse
        Returns V_light value (N,) and grad_V_light (N, D).
        """
        N_dot_L = np.sum(grad_phi * light_dir, axis=-1)

        # C1 Smoothplus approximation of max(0, N.L)
        eps = 1e-3
        I_diffuse = 0.5 * (N_dot_L + np.sqrt(N_dot_L**2 + eps**2))

        # Compute soft shadows
        S = self.compute_soft_shadow(p, light_dir)

        V_light = -S * I_diffuse

        # Compute the spatial gradient of V_light using central differences
        # to ensure elegant, smooth vector flow
        eps_fd = 1e-2
        grad_V_light = np.zeros_like(p)
        D = p.shape[1]

        for i in range(D):
            eps_vec = np.zeros(D, dtype=np.float32)
            eps_vec[i] = eps_fd

            # Left shift
            p_l = p - eps_vec
            grad_phi_l = self.scene.evaluate_gradient(p_l)
            N_dot_L_l = np.sum(grad_phi_l * light_dir, axis=-1)
            I_diffuse_l = 0.5 * (N_dot_L_l + np.sqrt(N_dot_L_l**2 + eps**2))
            S_l = self.compute_soft_shadow(p_l, light_dir)
            V_light_l = -S_l * I_diffuse_l

            # Right shift
            p_r = p + eps_vec
            grad_phi_r = self.scene.evaluate_gradient(p_r)
            N_dot_L_r = np.sum(grad_phi_r * light_dir, axis=-1)
            I_diffuse_r = 0.5 * (N_dot_L_r + np.sqrt(N_dot_L_r**2 + eps**2))
            S_r = self.compute_soft_shadow(p_r, light_dir)
            V_light_r = -S_r * I_diffuse_r

            grad_V_light[:, i] = (V_light_r - V_light_l) / (2.0 * eps_fd)

        return V_light, grad_V_light

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        target: np.ndarray,
        light_dir: np.ndarray,
        dt: float,
        mass: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advances the physical system by one time step dt.
        Integrates Newtonian dynamics m * x'' = -∇U(x) and applies
        Inward-masked sliding projection to ensure condition-less, seamless boundary flow.
        """
        N, D = positions.shape
        # Normalize light direction
        light_dir_norm = light_dir / (np.linalg.norm(light_dir) + 1e-9)

        # 1. Evaluate scene geometry SDF & Gradient
        phi = self.scene.evaluate(positions)
        grad_phi = self.scene.evaluate_gradient(positions)

        # 2. Get Attractor Force (Huber-like)
        grad_V_attractor = self.evaluate_grad_V_attractor(positions, target)
        F_attractor = -self.w_attractor * grad_V_attractor

        # 3. Get Repulsive Barrier Force
        grad_V_collision = self.evaluate_grad_V_collision(phi, grad_phi)
        F_collision = -self.w_collision * grad_V_collision

        # 4. Get Light/Shadow Phototaxis Force
        _, grad_V_light = self.evaluate_V_light(positions, light_dir_norm, grad_phi)
        F_light = -self.w_light * grad_V_light

        # 5. Net Force and raw integration
        F_net = F_attractor + F_collision + F_light
        velocities_next = velocities + (F_net / mass) * dt

        # 6. Continuous Inward-masked Sliding Projection
        # Project velocity onto plane perpendicular to surface normal,
        # but only when moving inward (v . n < 0).
        v_dot_n = np.sum(velocities_next * grad_phi, axis=-1, keepdims=True)
        # Extract inward-moving component
        v_inward = np.minimum(0.0, v_dot_n) * grad_phi

        # Smooth boundary weight mask based on proximity to surface.
        # Active near or inside the boundary (phi <= collision_threshold)
        w_wall = smoothstep(self.collision_threshold, 0.0, phi)[:, np.newaxis]

        # Apply continuous projection without conditional branching
        velocities_slide = velocities_next - w_wall * v_inward

        # Integrate positions
        positions_next = positions + velocities_slide * dt

        return positions_next, velocities_slide
