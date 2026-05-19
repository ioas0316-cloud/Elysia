import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.Orchestra.conductor import Conductor, Instrument, Tempo, Mode, ElysiaCharter

def main():
    print("🌌 Initializing Unified Identity Test...")
    print(f"📖 Charter: {ElysiaCharter.get_essence()}")

    conductor = Conductor()
    conductor.register_instrument(Instrument("Reasoning", "Brain", lambda **k: "Thinking..."))

    # 1. Normal
    conductor.set_intent(mode=Mode.MAJOR)
    res = conductor.conduct_solo("Reasoning")
    print(f"Normal: {res}")

    # 2. Refusal
    conductor.set_intent(mode=Mode.MINOR, tempo=Tempo.ALLEGRO)
    res = conductor.conduct_solo("Reasoning")
    print(f"Refusal: {res}")

if __name__ == "__main__":
    main()
