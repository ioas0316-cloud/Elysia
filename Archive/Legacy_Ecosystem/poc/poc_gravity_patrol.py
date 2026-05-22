import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class KDFinalSecureUniverse:
    def __init__(self):
        # Master axes (Blue Core Frame)
        self.master_axes = [1+0j, 0+1j, -1+0j, 0-1j]
        self.cells = []

    def add_cell(self, cell_id, phase, category="Math", cell_type="normal"):
        self.cells.append({
            "id": cell_id,
            "phase": phase,
            "category": category,
            "type": cell_type,
            "status": "active"
        })

    def process_universal_stream(self):
        updated_cells = []
        logs = []
        for cell in self.cells:
            if cell["status"] == "removed":
                updated_cells.append(cell)
                continue

            phase_signal = cell["phase"]
            category = cell["category"]
            cell_id = cell["id"]

            # 1. Bandpass filter (Explosion! Red point -> poof!)
            if np.abs(phase_signal) > 1.5:
                logs.append(f"💥 [Bandpass Exceeded] Malicious attack '{cell_id}' destroyed!")
                cell["status"] = "removed"
                cell["type"] = "exploded"
                updated_cells.append(cell)
                continue

            # 2. Orthogonality check (Orange point -> 90 deg bounce)
            if category != "Math":
                orthogonal_trajectory = phase_signal * 1j
                logs.append(f"🍊 [Orthogonal Bounce] Category mismatch for '{cell_id}'. Bounced out!")
                cell["phase"] = orthogonal_trajectory
                # Make it move away from the center over time
                cell["phase"] *= 1.2
                if np.abs(cell["phase"]) > 2.0:
                     cell["status"] = "removed"
                updated_cells.append(cell)
                continue

            # 3. Normal / Isolated Cells Rescue
            # Find the closest master axis
            closest_axis = min(self.master_axes, key=lambda axis: np.abs(phase_signal - axis))
            angle_diff = np.angle(closest_axis) - np.angle(phase_signal)

            # Natural attraction (smooth drift) vs Forced Rescue (isolated)
            if np.abs(angle_diff) < 0.5:
                # Natural drift
                new_angle = np.angle(phase_signal) + angle_diff * 0.2
                cell["phase"] = np.abs(phase_signal) * np.exp(1j * new_angle)
                logs.append(f"🟢 [Natural Drift] '{cell_id}' smoothly drifting towards master axis.")
            else:
                # Forced rescue
                logs.append(f"🔵 [Forced Rescue] '{cell_id}' isolated! 'Why are you alone?' Forced snap to master axis!")
                cell["phase"] = closest_axis

            # Ensure it eventually settles on the circle radius 1
            magnitude_diff = 1.0 - np.abs(cell["phase"])
            cell["phase"] = (np.abs(cell["phase"]) + magnitude_diff * 0.2) * np.exp(1j * np.angle(cell["phase"]))

            updated_cells.append(cell)

        self.cells = updated_cells
        return logs

engine = KDFinalSecureUniverse()

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
ax.set_title("KD Final Secure Universe: Global Homeostatic Patrol", fontsize=14)

# Plot Master Axes
master_x = [np.real(p) for p in engine.master_axes]
master_y = [np.imag(p) for p in engine.master_axes]
ax.scatter(master_x, master_y, color='blue', s=200, marker='X', label='Master Core (Phase Anchor)')

scat_normal = ax.scatter([], [], color='green', s=100, label='Clean Cell (Attraction)')
scat_isolated = ax.scatter([], [], color='purple', s=100, label='Isolated Cell (Rescue)')
scat_malicious = ax.scatter([], [], color='red', s=300, marker='*', label='Malicious (Explosion)')
scat_mismatch = ax.scatter([], [], color='orange', s=100, label='Mismatch (Bounced)')

ax.legend(loc='upper right', fontsize=10)

def init():
    scat_normal.set_offsets(np.empty((0, 2)))
    scat_isolated.set_offsets(np.empty((0, 2)))
    scat_malicious.set_offsets(np.empty((0, 2)))
    scat_mismatch.set_offsets(np.empty((0, 2)))
    return scat_normal, scat_isolated, scat_malicious, scat_mismatch

def update(frame):
    # Dynamically add noise/cells every few frames
    if frame % 10 == 0:
        cell_type = random.choice(["normal", "isolated", "malicious", "mismatch"])
        angle = random.uniform(0, 2 * np.pi)

        if cell_type == "normal":
            phase = 0.8 * np.exp(1j * angle)
            engine.add_cell(f"Cell_{frame}", phase, "Math", "normal")
        elif cell_type == "isolated":
            # Completely misaligned angle
            phase = 1.0 * np.exp(1j * (angle + np.pi/4))
            engine.add_cell(f"Iso_{frame}", phase, "Math", "isolated")
        elif cell_type == "malicious":
            # Huge amplitude
            phase = 1.8 * np.exp(1j * angle)
            engine.add_cell(f"Mal_{frame}", phase, "Math", "malicious")
        elif cell_type == "mismatch":
            phase = 1.0 * np.exp(1j * angle)
            engine.add_cell(f"Mis_{frame}", phase, "Cooking", "mismatch")

    logs = engine.process_universal_stream()
    if logs:
        print(f"--- Frame {frame} ---")
        for log in logs:
            print(log)

    normal_coords = []
    isolated_coords = []
    malicious_coords = []
    mismatch_coords = []

    for cell in engine.cells:
        if cell["status"] == "removed":
            continue

        x, y = np.real(cell["phase"]), np.imag(cell["phase"])
        if cell["type"] == "normal":
            normal_coords.append([x, y])
        elif cell["type"] == "isolated":
            isolated_coords.append([x, y])
        elif cell["type"] == "malicious":
            malicious_coords.append([x, y])
        elif cell["type"] == "mismatch":
            mismatch_coords.append([x, y])

    scat_normal.set_offsets(np.array(normal_coords) if normal_coords else np.empty((0, 2)))
    scat_isolated.set_offsets(np.array(isolated_coords) if isolated_coords else np.empty((0, 2)))
    scat_malicious.set_offsets(np.array(malicious_coords) if malicious_coords else np.empty((0, 2)))
    scat_mismatch.set_offsets(np.array(mismatch_coords) if mismatch_coords else np.empty((0, 2)))

    return scat_normal, scat_isolated, scat_malicious, scat_mismatch

ani = animation.FuncAnimation(fig, update, frames=60, init_func=init, blit=True, interval=200)
ani.save('poc/gravity_patrol_simulation.gif', writer='pillow', fps=5)
plt.close()
print("Simulation complete. GIF saved to poc/gravity_patrol_simulation.gif")
