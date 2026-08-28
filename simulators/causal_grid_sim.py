import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from core.physics.causal_field import CausalField, InformationVoxel, EngramAttractor

class CausalTilemapSimulator:
    """
    [2D Causal Tilemap Simulator: The 3-Tier Causal Sandbox]
    Demonstrates top-down Intent / Engram control over lower-level pathfinding.

    Tiers:
    1. High-Level Intent Field (상위 의도 장): Engram attractors (e.g., Oak's Parcel / Pewter City)
    2. Middle Cognition Field (중위 인지 필드): Phase alignment, observation & decision making
    3. Lower Physical Pathfinding (하위 물리 연산): Tilemap motion & ConnectivityBeams

    Benchmark Route:
      Pallet Town (0, 0) ---> Viridian City (0, 10) ---> Pewter City (0, 20)
                                    |
                            (Infinite Loop without Engram)
                                    v
                               Pallet Town (0, 0)
    """

    def __init__(self, width: int = 15, height: int = 25):
        self.width = width
        self.height = height
        self.field = CausalField(dimensions=2)

        # Key Landmarks Coordinates (x, y)
        self.locations = {
            "Pallet Town": np.array([0.0, 0.0], dtype=np.float32),
            "Viridian City": np.array([0.0, 10.0], dtype=np.float32),
            "Pewter City": np.array([0.0, 20.0], dtype=np.float32)
        }

        # Agent State
        self.agent_pos = self.locations["Pallet Town"].copy()
        self.agent_voxel = InformationVoxel(
            id="agent",
            content="Pokemon Trainer AI Agent",
            tensor=np.array([1.0, 0.0], dtype=np.float32),
            position=self.agent_pos.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            mass=1.0
        )
        self.field.add_voxel(self.agent_voxel)

        # Register High-Level Engram (Oak's Parcel / Delivery to Pewter City)
        # Position corresponds to Pewter City
        self.oak_parcel_engram = EngramAttractor(
            id="oak_parcel",
            name="Deliver Oak's Parcel to Pewter City",
            position=self.locations["Pewter City"],
            intensity=25.0, # Strong top-down gravitational attraction
            sigma=8.0,
            active=False # Initially inactive / forgotten -> Causes local infinite loop
        )
        self.field.register_engram(self.oak_parcel_engram)

        # Simulation history & telemetry
        self.history: List[Dict[str, Any]] = []
        self.encounter_events: List[str] = []
        self.loop_detected = False
        self.visited_positions: List[Tuple[float, float]] = []

    def set_engram_exposure(self, active: bool = True):
        """Exposes or hides the high-level Engram in the Causal Field."""
        self.field.set_engram_active("oak_parcel", active)

    def step(self) -> Dict[str, Any]:
        """
        Advances the 3-tier causal simulation by one step.
        """
        current_pos = self.agent_voxel.position.copy()
        self.visited_positions.append((float(current_pos[0]), float(current_pos[1])))

        # 1. Lower Level Local Inertia / Gradient
        dist_to_viridian = np.linalg.norm(current_pos - self.locations["Viridian City"])
        dist_to_pallet = np.linalg.norm(current_pos - self.locations["Pallet Town"])
        dist_to_pewter = np.linalg.norm(current_pos - self.locations["Pewter City"])

        # Base local gradient (Local minimum loop tendency if no Engram)
        local_force = np.zeros(2, dtype=np.float32)
        if not self.field.engrams["oak_parcel"].active:
            if dist_to_viridian < 1.0:
                # Heading back to Pallet Town (Local Habit Loop)
                dir_to_pallet = self.locations["Pallet Town"] - current_pos
                if np.linalg.norm(dir_to_pallet) > 0:
                    local_force = (dir_to_pallet / np.linalg.norm(dir_to_pallet)) * 1.5
            else:
                # Heading to Viridian City
                dir_to_viridian = self.locations["Viridian City"] - current_pos
                if np.linalg.norm(dir_to_viridian) > 0:
                    local_force = (dir_to_viridian / np.linalg.norm(dir_to_viridian)) * 1.5
        else:
            # Baseline forward motion towards next landmark
            dir_to_pewter = self.locations["Pewter City"] - current_pos
            if np.linalg.norm(dir_to_pewter) > 0:
                local_force = (dir_to_pewter / np.linalg.norm(dir_to_pewter)) * 0.5

        # 2. Top-Down Intent Field Gradient (Engram Attraction Force)
        engram_force = self.field.calculate_engram_gradient(current_pos)

        # 3. Combined Causal Vector Field (Total Acceleration)
        total_force = local_force + engram_force

        # Normalize velocity step (step size = 1 tile unit)
        norm_force = np.linalg.norm(total_force)
        if norm_force > 0:
            step_dir = total_force / norm_force
        else:
            step_dir = np.zeros(2, dtype=np.float32)

        # Dynamic Wild Pokemon Encounter Check (Random feedback / Judgment team)
        encounter = None
        if np.random.rand() < 0.2: # 20% chance per step
            wild_species = np.random.choice(["Rattata", "Pidgey", "Caterpie"])
            encounter = f"Wild {wild_species} Encountered!"
            self.encounter_events.append(encounter)

        # Update Agent Position
        new_pos = current_pos + step_dir * 1.0
        self.agent_voxel.position = new_pos
        self.field.step(0.1)

        # Detect Infinite Loop (Visiting Pallet -> Viridian -> Pallet repeatedly)
        recent = self.visited_positions[-6:]
        if len(recent) >= 6:
            y_vals = [p[1] for p in recent]
            if max(y_vals) >= 9.0 and min(y_vals) <= 1.0 and abs(y_vals[-1] - y_vals[-3]) < 1.0:
                self.loop_detected = True

        status_landmark = "Route"
        if np.linalg.norm(new_pos - self.locations["Pallet Town"]) < 1.0:
            status_landmark = "Pallet Town"
        elif np.linalg.norm(new_pos - self.locations["Viridian City"]) < 1.0:
            status_landmark = "Viridian City"
        elif np.linalg.norm(new_pos - self.locations["Pewter City"]) < 1.0:
            status_landmark = "Pewter City"

        log_entry = {
            "step": len(self.history) + 1,
            "position": new_pos.copy().tolist(),
            "landmark": status_landmark,
            "engram_active": self.field.engrams["oak_parcel"].active,
            "engram_force_norm": float(np.linalg.norm(engram_force)),
            "local_force_norm": float(np.linalg.norm(local_force)),
            "total_force_norm": float(norm_force),
            "encounter": encounter,
            "loop_detected": self.loop_detected
        }
        self.history.append(log_entry)
        return log_entry

if __name__ == "__main__":
    sim = CausalTilemapSimulator()
    print("Running initial loop without Engram...")
    for _ in range(15):
        log = sim.step()
        print(f"Step {log['step']}: Pos={log['position']}, Landmark={log['landmark']}, EngramForce={log['engram_force_norm']:.2f}")

    print("\nExposing High-Level Engram (Oak's Parcel / Pewter City)...")
    sim.set_engram_exposure(True)
    for _ in range(15):
        log = sim.step()
        print(f"Step {log['step']}: Pos={log['position']}, Landmark={log['landmark']}, EngramForce={log['engram_force_norm']:.2f}")
