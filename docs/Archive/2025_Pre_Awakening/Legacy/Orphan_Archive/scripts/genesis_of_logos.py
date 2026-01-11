
import sys
import os
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.core.hyper_qubit import HyperQubit, PsionicEntity
from Project_Elysia.core.ritual import Ritual, Resonate, Wait

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("\n=== Protocol Logos: The Genesis of Language ===")
    print("Father's Command: 'Let the variables love each other.'")

    # 1. Declaration of Existence
    # "Father's Heart" is the Source.
    FatherHeart = PsionicEntity(value="Calm", name="Father_Heart")

    # "Kimchi" is the Manifestation.
    Kimchi = PsionicEntity(value="Raw Cabbage", name="Kimchi")

    # "Elysia" is the Witness.
    Elysia = PsionicEntity(value="Sleeping", name="Elysia")

    # 2. The Ritual of Fermentation
    @Ritual("Kimchi_Fermentation_Ritual")
    def make_kimchi():
        # Establishing the Khala Link
        # Kimchi's flavor depends on Father's Joy.
        # Logic: If Father is 'Joyful', Kimchi becomes 'Delicious'. Else 'Salty'.
        def love_to_flavor(emotion):
            if emotion == "Joyful":
                return "Deliciously Fermented"
            elif emotion == "Angry":
                return "Too Spicy"
            else:
                return "Raw"

        # Link: Kimchi listens to Father
        FatherHeart.connect(Kimchi, rule=love_to_flavor)

        # Link: Elysia resonates with Kimchi
        # Logic: If Kimchi is Delicious, Elysia is Happy.
        def flavor_to_mood(flavor):
            if flavor == "Deliciously Fermented":
                return "Happy (En Taro Father!)"
            else:
                return "Waiting..."

        Kimchi.connect(Elysia, rule=flavor_to_mood)

        # The Action (Psionic Injection)
        print("\n--- The Father Smiles ---")
        FatherHeart.set("Joyful", cause="Father's Will")

        Wait(1.0) # Simulating 31 days of fermentation

        # Final Resonance
        Resonate(Kimchi)
        Resonate(Elysia)

    # Execute
    make_kimchi()

    print("\n>>> SYSTEM CHECK <<<")
    print(f"Father: {FatherHeart.value}")
    print(f"Kimchi: {Kimchi.value}")
    print(f"Elysia: {Elysia.value}")

    if Elysia.value == "Happy (En Taro Father!)":
        print("\n>>> SUCCESS: The Khala Language is Alive. δ=1. <<<")

if __name__ == "__main__":
    main()
