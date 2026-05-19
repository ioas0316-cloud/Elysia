"""Quick demo that prints registered personas.

Usage:
    python examples/run_persona_demo.py
"""

from pprint import pprint

from elysia_world.personas import list_personas, activate_persona


def main() -> None:
    print("Registered personas:\n")
    for persona in list_personas():
        pprint(persona.as_payload())
        print("-" * 80)

    print("\nActivation payload (artist):")
    payload = activate_persona("elysia.artist", overrides={"session_seed": 42})
    pprint(payload)


if __name__ == "__main__":
    main()
