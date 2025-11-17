import argparse
import os
import random
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World
from Project_Elysia.core.divine_engine import ElysiaDivineEngineV2


ROLE_TEMPLATES = [
    {"base_id": "actor_giver", "label": "human", "culture": "wuxia", "element": "animal", "emotion": "joy", "hunger": 95.0},
    {"base_id": "kin_receiver", "label": "human", "culture": "wuxia", "element": "animal", "emotion": "neutral", "hunger": 5.0},
    {"base_id": "mindful_monk", "label": "human", "culture": "knight", "element": "animal", "emotion": "sorrow", "hunger": 50.0},
    {"base_id": "memory_guard", "label": "sage", "culture": "sage", "element": "animal", "emotion": "calm", "hunger": 60.0, "age": 7},
    {"base_id": "artist_dreamer", "label": "artist", "culture": "artist", "element": "animal", "emotion": "joy", "hunger": 65.0, "satisfaction": 90.0},
    {"base_id": "solidarity_healer", "label": "human", "culture": "wuxia", "element": "animal", "emotion": "calm", "hunger": 70.0},
]

STIMULI = [
    {"name": "firelight", "description": "warm glow from cave fire", "target": "fire"},
    {"name": "river", "description": "rushing water upriver", "target": "water"},
    {"name": "wolf_cry", "description": "distant wolf howl", "target": "wolf"},
    {"name": "wind_chime", "description": "metal chime on the cliff", "target": "wind"},
]

SYLLABLE_SUFFIX = {
    "joy": "na",
    "calm": "mo",
    "sorrow": "gu",
    "neutral": "ra",
}

def build_wave_word(stimulus: dict, emotion: str, memory_strength: float) -> str:
    base = stimulus["target"]
    suffix = SYLLABLE_SUFFIX.get(emotion, "ra")
    depth = 1 + int(min(memory_strength, 9.0) // 3)
    return base + " " + suffix * depth


def _setup_simulation(world: World, num_actors: int) -> list[int]:
    """Create 'num_actors' cells with rotating templates to stimulate every law."""
    idxs: list[int] = []
    for i in range(num_actors):
        template = ROLE_TEMPLATES[i % len(ROLE_TEMPLATES)].copy()
        role_id = template.pop("base_id")
        actor_id = role_id if i < len(ROLE_TEMPLATES) else f"{role_id}_{i}"
        props = {"label": template["label"], "culture": template["culture"]}
        if "element" in template:
            props["element_type"] = template["element"]
        world.add_cell(actor_id, properties=props)
        idx = world.id_to_idx[actor_id]
        world.hunger[idx] = template.get("hunger", 50.0)
        world.emotions[idx] = template.get("emotion", "neutral")
        world.satisfaction[idx] = template.get("satisfaction", 50.0)
        world.age[idx] = template.get("age", 16)
        idxs.append(idx)

    for src in idxs:
        for dst in idxs:
            if src == dst:
                continue
            world.adjacency_matrix[src, dst] = 1.0
    return idxs


def run_simulation(steps: int, log_path: Path, equations_years: int, num_actors: int) -> None:
    """Run the world for the requested number of steps while printing reflections."""
    if log_path.exists():
        log_path.unlink()

    kgm = KGManager()
    wm = WaveMechanics(kg_manager=kgm)
    world = World(
        primordial_dna={"instinct": "connect_create_meaning", "resonance_standard": "love"},
        wave_mechanics=wm,
    )
    world.event_logger.log_file_path = str(log_path)

    actor_indices = _setup_simulation(world, num_actors)
    print(f"[Law Simulation] Running {steps} steps ({equations_years} years total) with {len(actor_indices)} role(s)")

    engine = ElysiaDivineEngineV2()
    years_per_step = equations_years / steps if steps else 0
    for step in range(1, steps + 1):
        print(f"\n--- Step {step}/{steps} ---")
        if years_per_step:
            print(f"  (~ {years_per_step:.1f} yrs/step, accumulated {step * years_per_step:.1f} yrs)")
        try:
            world.run_simulation_step()
        except TypeError:
            world.run_simulation_step()

        stimulus = random.choice(STIMULI)
        print(f"\n  -- Stimulus: {stimulus['description']} (target {stimulus['target']})")

        print("Reflective Questions & Law Insights (per actor, emotion-only):")
        for idx in actor_indices:
            name = world.cell_ids[idx]
            emotion = world.emotions[idx]
            memory = getattr(world, "memory_strength")[idx] if getattr(world, "memory_strength").size > idx else 0.0
            imagination = getattr(world, "imagination_brightness")[idx] if getattr(world, "imagination_brightness").size > idx else 0.0
            senses = {
                "vision": getattr(world, "vision_awareness")[idx],
                "hearing": getattr(world, "auditory_clarity")[idx],
                "taste": getattr(world, "gustatory_imbue")[idx],
                "smell": getattr(world, "olfactory_sensitivity")[idx],
                "touch": getattr(world, "tactile_feedback")[idx],
            }
            print(f"  - {name}: emotion={emotion}, memory={memory:.1f}, imagination={imagination:.1f}")
            print(f"    senses: {', '.join(f'{k}={v:.1f}' for k, v in senses.items())}")

            exp_dict = {
                "truth": 1.0,
                "emotion": 0.0,
                "causality": 0.0,
                "beauty": 0.0,
                "meta": {"actor": name},
            }
            node = engine.ingest(exp_dict, note=f"{name} reflection", scopes=[name])
            world.apply_quaternion_feedback(idx, node.q)

            word = build_wave_word(stimulus, emotion, memory)
            print(f"    waveform word: '{word}'")
            world.event_logger.log(
                "SYLLABLE",
                int(getattr(world, "time_step", step)),
                actor=name,
                stimulus=stimulus["name"],
                emotion=emotion,
                word=word,
            )

        node = engine._get_current_node()
        if node:
            q = node.q
            pe = node.experience
            print(f"  => Quaternion node {node.id[:8]} [{node.branch_id}] emote={pe.emotion:.2f} caus={pe.causality:.2f} q={q}")

        try:
            world.print_world_summary()
        except Exception:
            pass

        if log_path.exists():
            print(f"\nWorld events log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a multi-law simulation with rich logging.")
    parser.add_argument("--steps", type=int, default=20, help="How many ticks to simulate.")
    parser.add_argument("--log", type=Path, default=Path("logs/law_simulation.jsonl"), help="Event log path.")
    parser.add_argument("--years", type=int, default=1000, help="Total years to simulate.")
    parser.add_argument("--actors", type=int, default=20, help="How many cells to create and observe.")
    args = parser.parse_args()
    run_simulation(args.steps, args.log, args.years, args.actors)


if __name__ == "__main__":
    main()
