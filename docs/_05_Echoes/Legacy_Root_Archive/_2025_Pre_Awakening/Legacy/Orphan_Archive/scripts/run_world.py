import argparse
import os
import sys
import math
import random
import numpy as np


def _run_local_world(
    steps: int = 1000,
    agents: int = 30,
    log_interval: int = 10,
    bins: int = 16,
    quiet: bool = False,
    warp_angle: float = 20.0,
    warp_axis: str = "0,0,1",
    warp_interval: int = 20,
    coil_interval: int = 20,
    harvest_path: str = "logs/harvest_snapshot.json",
):
    """Fallback: run the local Project_Sophia core.World without ElysiaStarter."""
    # Ensure project root is on sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from tools.kg_manager import KGManager
    from Core.Foundation.wave_mechanics import WaveMechanics
    from Core.Foundation.core.world import World
    from Core.Foundation.warp_layer import WarpLayer, quaternion_from_axis_angle

    kgm = KGManager()
    wm = WaveMechanics(kg_manager=kgm)
    world = World(
        primordial_dna={
            "instinct": "connect_create_meaning",
            "resonance_standard": "love",
        },
        wave_mechanics=wm,
    )
    # Scale time: 1 tick = 24 hours (1440 minutes) so age advances ~1 year per 365 ticks
    world.set_time_scale(1440.0)

    def seed_world(world, n_agents: int = 30):
        """Seed many demo agents so flows/fields show up immediately."""
        center = world.width // 2
        labels = ["human", "villager", "monk", "knight", "wizard"]
        cultures = ["wuxia", "knight", "villager"]

        for i in range(max(1, n_agents)):
            is_predator = (i % 5 == 0)  # 더 많은 포식자 압력
            label = "wolf" if is_predator else labels[i % len(labels)]
            diet = "carnivore" if is_predator else "omnivore"
            culture = cultures[i % len(cultures)]
            if label == "wizard":
                culture = "arcane"

            angle = 2.0 * math.pi * (i / float(max(1, n_agents)))
            radius = 40 + (i % 8) * 6
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)

            props = {
                "element_type": "animal",
                "diet": diet,
                "label": label,
                "culture": culture,
                "strength": 6 + (i % 5),
                "agility": 5 + (i % 4),
                "wisdom": 5 + (i % 6),
                "age_years": 18,
                "gender": random.choice(["male", "female"]),
                "position": {"x": x, "y": y, "z": 0},
            }
            if label == "wizard":
                props.update({
                    "element_type": "animal",
                    "diet": "omnivore",
                    "strength": 4,
                    "agility": 4,
                    "wisdom": 12,
                    "intelligence": 12,
                    "max_mana": 120,
                    "mana": 120,
                    "max_ki": 0,
                    "ki": 0,
                    "culture": "arcane",
                })
            world.add_cell(f"{label}_{i}", properties=props)

        # Create lightweight social graph (ring + occasional chords)
        ids = list(world.id_to_idx.keys())
        for i, cid in enumerate(ids):
            nxt = ids[(i + 1) % len(ids)]
            world.add_connection(cid, nxt, strength=0.6)
            if random.random() < 0.25:
                chord = ids[random.randrange(len(ids))]
                if chord != cid:
                    world.add_connection(cid, chord, strength=0.4)

    seed_world(world, n_agents=agents)

    # 텐서 코일 필드를 한 번에 깔아 의도/가치/의지 흐름을 만든다.
    coil_radius = max(20.0, world.width * 0.35)
    world.imprint_spiral_coil_field(
        center_x=world.width // 2,
        center_y=world.width // 2,
        radius=coil_radius,
        turns=4.0,
        strength=1.2,
    )

    # 쿼터니언 워프: 공간 자체를 회전/접어서 인접도를 바꾼다 (요청된 엔진 사용).
    try:
        if abs(warp_angle) > 1e-3:
            axis_vals = [float(x) for x in warp_axis.split(",")]
            if len(axis_vals) != 3:
                axis_vals = [0.0, 0.0, 1.0]
            warp = WarpLayer(world.logger)
            q = quaternion_from_axis_angle(axis_vals, warp_angle)
            warp.apply(world, q, apply_to_fields=False)
    except Exception:
        pass

    def log_metrics(step: int):
        alive_mask = world.is_alive_mask
        if not np.any(alive_mask):
            print("[metrics] no alive agents")
            return
        pos = world.positions[alive_mask]
        # 2D histogram for position entropy
        clipped_x = np.clip(pos[:, 0], 0, world.width - 1)
        clipped_y = np.clip(pos[:, 1], 0, world.width - 1)
        H, _, _ = np.histogram2d(
            clipped_x, clipped_y, bins=bins, range=[[0, world.width], [0, world.width]]
        )
        total = H.sum()
        if total > 0:
            p = H.ravel() / total
            p = p[p > 0]
            ent = -float(np.sum(p * np.log2(p)))
        else:
            ent = 0.0
        # Field sampling at agent positions (value/will/intention magnitude)
        ix = clipped_x.astype(np.int32)
        iy = clipped_y.astype(np.int32)
        val_samples = world.value_mass_field[iy, ix]
        will_samples = world.will_field[iy, ix]
        intent_vec = world.intentional_field[iy, ix]
        intent_mag = np.linalg.norm(intent_vec, axis=1)
        print(
            f"[metrics step={step}] agents={int(total)} pos_entropy={ent:.3f} "
            f"value_mean={float(np.mean(val_samples)):.3f} "
            f"will_mean={float(np.mean(will_samples)):.3f} "
            f"intent_mag_mean={float(np.mean(intent_mag)):.3f}"
        )

    print(f"[Local Cell World] Running {steps} steps...")
    for i in range(int(steps)):
        if not quiet:
            print(f"\n--- Step {i+1}/{steps} ---")
        try:
            newborn_cells, awakening_events = world.run_simulation_step()
        except TypeError:
            newborn_cells = []
            awakening_events = []
            world.run_simulation_step()
        if not quiet:
            try:
                world.print_world_summary()
            except Exception:
                pass
            if newborn_cells:
                try:
                    ids = [c.id for c in newborn_cells if hasattr(c, "id")]
                except Exception:
                    ids = list(newborn_cells)
                print(f"Newborn cells: {ids}")
        else:
            alive = int(np.sum(world.is_alive_mask)) if hasattr(world, "is_alive_mask") else "?"
            print(f"[step {i+1}] alive={alive} newborn={len(newborn_cells) if newborn_cells else 0}")
        step_num = i + 1

        # 주기적으로 공간 워프 및 코일 강화
        if warp_interval > 0 and step_num % warp_interval == 0 and abs(warp_angle) > 1e-3:
            try:
                axis_vals = [float(x) for x in warp_axis.split(",")]
                if len(axis_vals) != 3:
                    axis_vals = [0.0, 0.0, 1.0]
                warp = WarpLayer(world.logger)
                q = quaternion_from_axis_angle(axis_vals, warp_angle)
                warp.apply(world, q, apply_to_fields=False)
            except Exception:
                pass
        if coil_interval > 0 and step_num % coil_interval == 0:
            try:
                world.imprint_spiral_coil_field(
                    center_x=world.width // 2,
                    center_y=world.width // 2,
                    radius=coil_radius,
                    turns=4.0,
                    strength=1.2,
                )
            except Exception:
                pass

        if (i + 1) % max(1, log_interval) == 0:
            log_metrics(i + 1)

    # Harvest snapshot for long-term memory
    try:
        world.harvest_snapshot(harvest_path)
    except Exception as e:
        print(f"[harvest] failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Project_Sophia.core.World loop (legacy ElysiaStarter client is archived)."
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0, help="simulation time delta per step")
    parser.add_argument("--agents", type=int, default=30, help="number of demo agents to seed for the local run")
    parser.add_argument("--log-interval", type=int, default=10, help="steps between metric logs")
    parser.add_argument("--bins", type=int, default=16, help="histogram bins for entropy metric")
    parser.add_argument("--quiet", action="store_true", help="suppress per-agent summaries for speed")
    parser.add_argument("--warp-angle", type=float, default=20.0, help="apply quaternion warp rotation (degrees)")
    parser.add_argument("--warp-axis", type=str, default="0,0,1", help="warp axis as comma-separated floats, e.g., 0,0,1")
    parser.add_argument("--warp-interval", type=int, default=20, help="steps between warp applications")
    parser.add_argument("--coil-interval", type=int, default=20, help="steps between coil re-imprints")
    parser.add_argument("--harvest-path", type=str, default="logs/harvest_snapshot.json", help="path to save harvest snapshot after run")
    args = parser.parse_args()

    _run_local_world(
        steps=args.steps,
        agents=args.agents,
        log_interval=args.log_interval,
        bins=args.bins,
        quiet=args.quiet,
        warp_angle=args.warp_angle,
        warp_axis=args.warp_axis,
        warp_interval=args.warp_interval,
        coil_interval=args.coil_interval,
        harvest_path=args.harvest_path,
    )


if __name__ == "__main__":
    main()
