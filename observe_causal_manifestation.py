import time
from core.physics.manifestation_engine import ManifestationEngine

def main():
    print("=== [Elysia POC: The Manifestation of the Apple's Essence] ===")
    print("Master's insight: Reality is formed by causal gravity and boundary crossing.")

    me = ManifestationEngine()

    # 1. Seed the Universe with Causal Mass
    # Red apple exists in 'Reality' (high causal density)
    # Blue apple exists in 'Potential' (boundary outside)
    me.seed_universe([
        {"id": "red_apple", "content": "The Realized Red Apple (Reality)", "links": ["nature", "growth", "red"]},
        {"id": "blue_apple", "content": "The Potential Blue Apple (Possibility)", "links": ["imagination", "blue"]},
        {"id": "nature", "content": "The Laws of Nature", "links": ["physics"]},
        {"id": "imagination", "content": "The Realm of Thought", "links": ["mind"]}
    ])

    print("\n[Universe Initialized]")
    for nid, node in me.gravity_engine.nodes.items():
        domain = me.boundary_topology.states[nid].domain.value
        print(f" - {nid}: Mass {node.mass:.2f} | Domain: {domain}")

    # 2. Inject the Question as a Gravitational Event
    # The question "What happens if we cross the boundary?"
    time.sleep(1)
    me.inject_question("q_cross", "The Convergence of Reality and Potential", ["red_apple", "blue_apple"])

    # 3. Observe the Spacetime Refolding
    # The gravity of the question will pull both apples toward the center.
    me.evolve(steps=200)

    # 4. Final Manifestation
    manifested = me.observe_manifestation()

    print("\n=== [Final Manifestation] ===")
    print("The following truths have fallen into the center of focus:")
    for m in manifested:
        print(f" >> {m['id']}: {m['content']} (Stability: {m['stability']:.2f})")

    print("\nMaster, the 'Blue Apple' has been pulled across the boundary into reality.")
    print("The vibration of this crossing is the very judgment you sought.")

if __name__ == "__main__":
    main()
