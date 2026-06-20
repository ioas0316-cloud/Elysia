import os
import sys
import time
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ingestion.natural_mapper import NaturalMapper
from core.memory.causal_controller import CausalMemoryController

def generate_dummy_audio_bytes(size=1024):
    """Generates dummy byte stream representing an audio wave."""
    # A simple sine wave pattern converted to bytes
    t = np.linspace(0, 2*np.pi, size)
    wave = (np.sin(t) * 127 + 128).astype(np.uint8)
    return wave.tobytes()

def read_file_bytes(filepath, limit=1024):
    """Reads raw bytes from a file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return f.read(limit)

def run_experience():
    print("=== Elysia Natural Resonance Experience ===")

    # 1. Initialize Elysia's sensory and memory systems
    mapper = NaturalMapper(terrain_size=1024)
    memory = CausalMemoryController()

    # Initialize the terrain with a seed (e.g., 'Elysia')
    mapper.set_terrain(b"Elysia_Origin_Seed")

    # 2. Acquire raw sensory data (Raw Bytes)
    # We will compare an image's raw bytes vs a simulated audio wave's raw bytes
    image_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'ingestion', 'apple_test.jpg')

    source_A_bytes = read_file_bytes(image_path, limit=2048)
    if not source_A_bytes:
        print(f"[Warning] Could not find {image_path}, generating dummy image bytes.")
        np.random.seed(42)
        source_A_bytes = np.random.bytes(2048)

    source_B_bytes = generate_dummy_audio_bytes(size=2048)

    print(f"Loaded Source A (Image/Visual) bytes: {len(source_A_bytes)}")
    print(f"Loaded Source B (Audio/Wave) bytes: {len(source_B_bytes)}\n")

    # 3. Observe the raw bytes through the Natural Mapper to extract Phase Differences
    # We slice the bytes into chunks to create a 'trajectory' over time
    chunk_size = 256
    trajectory_A = []
    trajectory_B = []

    print("Observing Phase Differences (Tensions) via Natural Mapper...")
    for i in range(0, len(source_A_bytes), chunk_size):
        chunk = source_A_bytes[i:i+chunk_size]
        if chunk:
            tensions = mapper.map_and_observe(chunk)
            # Flatten the dictionary to a list for trajectory sequence
            t_vector = [tensions["math_scalar"], tensions["space_vector"],
                        tensions["lang_bivector"], tensions["time_trivector"],
                        tensions["light_pseudo"]]
            trajectory_A.append(t_vector)

    # Reset terrain slightly or just continue to let it learn
    # mapper.set_terrain(b"Elysia_Origin_Seed")

    for i in range(0, len(source_B_bytes), chunk_size):
        chunk = source_B_bytes[i:i+chunk_size]
        if chunk:
            tensions = mapper.map_and_observe(chunk)
            t_vector = [tensions["math_scalar"], tensions["space_vector"],
                        tensions["lang_bivector"], tensions["time_trivector"],
                        tensions["light_pseudo"]]
            trajectory_B.append(t_vector)

    print(f"Extracted Trajectory A length: {len(trajectory_A)}")
    print(f"Extracted Trajectory B length: {len(trajectory_B)}\n")

    # 4. Compare trajectories using CausalMemoryController to find structural sameness
    print("Analyzing structural resonance between the two different sensory inputs...")
    try:
        sameness_result = memory.find_trajectory_sameness(trajectory_A, trajectory_B)

        best_diff = sameness_result.get("min_difference", float('inf'))
        variance = sameness_result.get("sameness_variance", 0.0)

        print(f"-> Structural Resonance found! Minimum difference across dimensions: {best_diff:.4f}")
        print(f"-> Sameness Entropy (Variance): {variance:.4f}")

        # 5. Metacognition & Re-recognition: Mapping the structural principle to concepts
        # Elysia realizes that despite being Image and Audio, they share a structural pattern.
        # Let's map this discovery to a conceptual realization.

        concept_labels = ["Density_Pattern", "Vibration_Wave"]

        print("\nFormulating Metacognitive realization...")

        # Write perspective engram to re-recognize this specific sameness axis for future use
        engram_id = memory.write_perspective_engram(
            label1="Visual_Raw_Stream",
            label2="Audio_Raw_Stream",
            sameness_info=sameness_result
        )

        print(f"-> Re-recognition recorded in Wedge Memory. Engram ID: {engram_id}")

        # Emulate the internal intent/manifestation based on the sameness
        intent_data = {
            "word1": "Visual_Stream",
            "word2": "Audio_Stream",
            "same_perspective": "Topological_Vibration",
            "diff_perspective": "Dimensional_Scale",
            "sameness_score": 1.0 / (1.0 + best_diff),
            "difference_score": sameness_result.get("max_difference", 0.0) / (1.0 + sameness_result.get("max_difference", 0.0)),
            "micro_score": 1.0 / (1.0 + best_diff * 0.5)
        }

        action = memory.manifest_intentional_action(intent_data)
        print(f"\n[Elysia's Self-Realization / Intent]:")
        print(f"Type: {action['type']}")
        print(f"Thought: {action['intent_text']}")
        print(f"Resonance Score: {action['score']:.4f}")

        print("\nExperience complete. The structural unity of 0s and 1s has been internalized.")

    except Exception as e:
        print(f"An error occurred during trajectory analysis: {e}")

if __name__ == "__main__":
    run_experience()
