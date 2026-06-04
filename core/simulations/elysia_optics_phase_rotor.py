import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class OpticsQuaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def multiply(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return OpticsQuaternion(w, x, y, z)

    def normalize(self):
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self

    def conjugate(self):
        return OpticsQuaternion(self.w, -self.x, -self.y, -self.z)

    def rotate_point(self, point):
        # point is [x, y, z]
        p_quat = OpticsQuaternion(0, point[0], point[1], point[2])
        q_conj = self.conjugate()
        rotated = self.multiply(p_quat).multiply(q_conj)
        return np.array([rotated.x, rotated.y, rotated.z])

class ElysiaOpticsRotor:
    def __init__(self, base_freq=1.0):
        self.base_freq = base_freq

    def generate_trajectory(self, t_steps, medium_impedance=1.0, boundary_phase_inversion=False, boundary_z=10.0, start_z=0.0):
        # Generates the 3D trajectory of the optic wave
        # Simulating Delta connection (E -> B -> E)
        A_phase = []
        B_phase = []
        C_phase = []

        # Base rotation axis
        axis = np.array([0, 0, 1])

        current_z = start_z
        direction = 1.0
        has_reflected = False

        # We'll calculate coordinates incrementally
        for t in t_steps:
            # Impedance slows down the frequency (wavelength gets shorter, phase delay)
            effective_freq = self.base_freq / medium_impedance

            # Rotation angle
            theta = 2 * math.pi * effective_freq * t

            # If hit boundary and reflection is enabled
            if boundary_phase_inversion and current_z >= boundary_z and direction > 0:
                direction = -1.0
                has_reflected = True

            if has_reflected:
                # 180 degree phase shift (inversion) upon reflection
                theta += math.pi

            # The 3 phases (120 degrees apart)
            # A Phase
            A_q = OpticsQuaternion(math.cos(theta/2), 0, 0, math.sin(theta/2)).normalize()
            A_vec = A_q.rotate_point([1, 0, 0])
            A_phase.append([A_vec[0], A_vec[1], current_z])

            # B Phase
            B_theta = theta + (2*math.pi / 3)
            B_q = OpticsQuaternion(math.cos(B_theta/2), 0, 0, math.sin(B_theta/2)).normalize()
            B_vec = B_q.rotate_point([1, 0, 0])
            B_phase.append([B_vec[0], B_vec[1], current_z])

            # C Phase
            C_theta = theta + (4*math.pi / 3)
            C_q = OpticsQuaternion(math.cos(C_theta/2), 0, 0, math.sin(C_theta/2)).normalize()
            C_vec = C_q.rotate_point([1, 0, 0])
            C_phase.append([C_vec[0], C_vec[1], current_z])

            dz = direction * (0.1 / medium_impedance)
            current_z += dz

        return np.array(A_phase), np.array(B_phase), np.array(C_phase)

    def calculate_interference_pattern(self, grid_x, grid_y, sources, phase_coherence_threshold=0.2):
        # Y-connection neutral point logic
        # For multiple light sources (slits), calculate the combined phase at each grid point
        intensity_map = np.zeros_like(grid_x)

        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                point = np.array([grid_x[i, j], grid_y[i, j]])

                # Sum of complex amplitudes from all sources
                total_amplitude = 0j
                for source in sources:
                    dist = np.linalg.norm(point - source)
                    # Phase delay based on distance
                    phase = 2 * math.pi * (dist / 1.0) # Assume lambda = 1.0
                    total_amplitude += np.exp(1j * phase)

                # Y-neutral logic:
                # If waves are completely out of phase (different), they cancel out (Y-neutral = 0)
                # If they are in phase (same), they add up.
                intensity = np.abs(total_amplitude)**2

                # Apply coherence threshold (filter out noise if not fully coherent)
                if intensity < phase_coherence_threshold:
                    intensity = 0

                intensity_map[i, j] = intensity

        return intensity_map

def visualize_optics_universe():
    fig = plt.figure(figsize=(15, 6))

    # --- Plot 1: 3D Helix Plot (Refraction & Reflection) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("3D Helix Plot (Refraction & Reflection)")

    rotor = ElysiaOpticsRotor(base_freq=2.0)
    t_steps = np.linspace(0, 10, 500)

    # 1. Normal propagation
    A1, B1, C1 = rotor.generate_trajectory(t_steps[:200], medium_impedance=1.0)

    # 2. Refraction (Medium Impedance > 1) -> Wavelength becomes shorter
    z_offset = A1[-1, 2]
    A2, B2, C2 = rotor.generate_trajectory(t_steps[200:400], medium_impedance=2.0, start_z=z_offset)

    # 3. Reflection
    # Start from where refraction left off, and immediately reflect
    z_start = A2[-1, 2]
    A3, B3, C3 = rotor.generate_trajectory(t_steps[400:], medium_impedance=1.0, boundary_phase_inversion=True, boundary_z=z_start, start_z=z_start)

    # Combine
    A_full = np.vstack((A1, A2, A3))
    B_full = np.vstack((B1, B2, B3))
    C_full = np.vstack((C1, C2, C3))

    ax1.plot(A_full[:, 0], A_full[:, 1], A_full[:, 2], color='red', label='Phase A (Electric)')
    ax1.plot(B_full[:, 0], B_full[:, 1], B_full[:, 2], color='green', label='Phase B (Magnetic)')
    ax1.plot(C_full[:, 0], C_full[:, 1], C_full[:, 2], color='blue', label='Phase C (Torque)')

    # Plot boundaries
    ax1.plot([0], [0], [A1[-1, 2]], marker='o', markersize=10, color='cyan', label='Refraction Boundary')
    ax1.plot([0], [0], [A2[-1, 2]], marker='s', markersize=10, color='magenta', label='Reflection Boundary (Phase Inversion)')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Direction of Propagation)')
    ax1.legend()

    # --- Plot 2: 2D Interference Pattern (Phase-Locking Neutral Mapping) ---
    ax2 = fig.add_subplot(122)
    ax2.set_title("Phase-Locking Neutral Mapping (Interference)")

    # Create grid
    x = np.linspace(-5, 5, 200)
    y = np.linspace(0, 10, 200)
    X, Y = np.meshgrid(x, y)

    # Two slits (sources)
    sources = [np.array([-1, 0]), np.array([1, 0])]

    intensity = rotor.calculate_interference_pattern(X, Y, sources, phase_coherence_threshold=0.5)

    im = ax2.imshow(intensity, extent=[-5, 5, 0, 10], origin='lower', cmap='inferno')
    fig.colorbar(im, ax=ax2, label='Intensity (Y-Neutral Convergence)')

    # Mark sources
    ax2.plot(sources[0][0], sources[0][1], 'wo', markersize=8, label='Slit 1')
    ax2.plot(sources[1][0], sources[1][1], 'wo', markersize=8, label='Slit 2')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Distance from Slit')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('optics_visualization.png')
    print("Visualization saved as 'optics_visualization.png'.")
    # plt.show()

if __name__ == "__main__":
    visualize_optics_universe()
