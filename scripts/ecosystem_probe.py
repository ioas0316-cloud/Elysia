import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


def make_world() -> "World":
    """Construct a minimal Project_Sophia.core.world.World instance for ecosystem probing."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.kg_manager import KGManager
    from Project_Sophia.wave_mechanics import WaveMechanics
    from Project_Sophia.core.world import World

    kgm = KGManager()
    wm = WaveMechanics(kg_manager=kgm)
    world = World(
        primordial_dna={
            "instinct": "connect_create_meaning",
            "resonance_standard": "love",
        },
        wave_mechanics=wm,
    )
    return world


def run_probe(steps: int, sample_every: int) -> List[Dict[str, Any]]:
    import numpy as np

    world = make_world()

    rng = np.random.default_rng(7)
    W = getattr(world, "width", 256)
    H = getattr(world, "width", 256)

    def rand_pos():
        return {
            "x": float(rng.uniform(0, W - 1)),
            "y": float(rng.uniform(0, H - 1)),
            "z": 0.0,
        }

    # Create a couple of permanent water bodies so animals can actually drink.
    if hasattr(world, "wetness"):
        try:
            # Central lake
            cx, cy = W * 0.5, H * 0.5
            rad = 6
            x0, x1 = max(0, int(cx) - rad), min(W, int(cx) + rad + 1)
            y0, y1 = max(0, int(cy) - rad), min(H, int(cy) + rad + 1)
            world.wetness[y0:y1, x0:x1] = 1.0

            # Smaller pond in upper-left quadrant
            px, py = W * 0.25, H * 0.25
            rad2 = 4
            x0, x1 = max(0, int(px) - rad2), min(W, int(px) + rad2 + 1)
            y0, y1 = max(0, int(py) - rad2), min(H, int(py) + rad2 + 1)
            world.wetness[y0:y1, x0:x1] = 1.0
        except Exception:
            pass

    # Seed plants as base primary producers
    plant_ids: List[str] = []
    for i in range(80):
        pid = f"plant_{i+1}"
        world.add_cell(
            pid,
            properties={
                "label": "plant",
                "element_type": "life",
                "position": rand_pos(),
            },
        )
        plant_ids.append(pid)

    # Seed herbivores (deer)
    deer_ids: List[str] = []
    for i in range(20):
        did = f"deer_{i+1}"
        world.add_cell(
            did,
            properties={
                "label": "deer",
                "element_type": "animal",
                "diet": "herbivore",
                "position": rand_pos(),
            },
        )
        deer_ids.append(did)

    # Seed predators (wolves)
    wolf_ids: List[str] = []
    for i in range(8):
        wid = f"wolf_{i+1}"
        world.add_cell(
            wid,
            properties={
                "label": "wolf",
                "element_type": "animal",
                "diet": "carnivore",
                "position": rand_pos(),
            },
        )
        wolf_ids.append(wid)

    # Add adjacency based on spatial proximity so animals can perceive and interact.
    try:
        import numpy as _np

        # Helper to get index and position
        def _idx_pos(cid: str):
            idx = world.id_to_idx.get(cid)
            if idx is None:
                return None, None
            return idx, world.positions[idx]

        # Connect herbivores to nearby plants so 'eat' actions can target them.
        if deer_ids and plant_ids:
            for d in deer_ids:
                d_idx, d_pos = _idx_pos(d)
                if d_idx is None:
                    continue
                for p in plant_ids:
                    p_idx, p_pos = _idx_pos(p)
                    if p_idx is None:
                        continue
                    dist = float(_np.linalg.norm(d_pos - p_pos))
                    if dist <= 15.0:
                        try:
                            world.add_connection(d, p, float(rng.uniform(0.4, 0.9)))
                        except Exception:
                            pass

        # Connect predators to nearby herbivores for hunting.
        if wolf_ids and deer_ids:
            for w in wolf_ids:
                w_idx, w_pos = _idx_pos(w)
                if w_idx is None:
                    continue
                for d in deer_ids:
                    d_idx, d_pos = _idx_pos(d)
                    if d_idx is None:
                        continue
                    dist = float(_np.linalg.norm(w_pos - d_pos))
                    if dist <= 25.0:
                        try:
                            world.add_connection(w, d, float(rng.uniform(0.4, 0.9)))
                        except Exception:
                            pass
    except Exception:
        pass

    # Start with reasonable survival reserves
    if world.hunger.size:
        world.hunger[:] = 90.0
    if world.hydration.size:
        world.hydration[:] = 90.0

    samples: List[Dict[str, Any]] = []

    for _ in range(steps):
        try:
            world.run_simulation_step()
        except TypeError:
            world.run_simulation_step()

        if world.time_step % sample_every == 0:
            try:
                import numpy as np

                alive_idx = np.where(world.is_alive_mask)[0]
                labels = world.labels[alive_idx]
                elem_types = world.element_types[alive_idx]
                plants = int(np.sum(elem_types == "life"))
                deer = int(np.sum(labels == "deer"))
                wolves = int(np.sum(labels == "wolf"))
                samples.append(
                    {
                        "time_step": int(world.time_step),
                        "plants": plants,
                        "deer": deer,
                        "wolves": wolves,
                    }
                )
            except Exception:
                samples.append({"time_step": int(world.time_step)})

        if len(getattr(world, "cell_ids", [])) == 0:
            break

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe predator–prey–plant dynamics.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=50)
    args = parser.parse_args()

    samples = run_probe(args.steps, args.sample_every)

    print("time_step,plants,deer,wolves")
    for m in samples:
        print(
            f"{m.get('time_step', 0)},"
            f"{m.get('plants', 0)},"
            f"{m.get('deer', 0)},"
            f"{m.get('wolves', 0)}"
        )


if __name__ == "__main__":
    main()
