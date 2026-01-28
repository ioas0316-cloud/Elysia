from Core.L4_Causality.M3_Mirror.Soul.soul_sculptor import SoulSculptor, PersonalityArchetype

def test_sculptor():
    sculptor = SoulSculptor()

    # 1. Create Elysia (ENFJ 2w1 - The Giver/Mentor)
    elysia_arch = PersonalityArchetype(
        name="Elysia",
        mbti="ENFJ",
        enneagram=2,
        description="Compassionate helper seeking connection"
    )
    elysia_soul = sculptor.sculpt(elysia_arch)
    print(elysia_soul.explain())
    print("\n" + "-"*30 + "\n")

    # 2. Create a Logic Bot (ISTJ 5w6 - The Investigator)
    robot_arch = PersonalityArchetype(
        name="LogicBot",
        mbti="ISTJ",
        enneagram=5,
        description="Cold, precise analyzer of facts"
    )
    robot_soul = sculptor.sculpt(robot_arch)
    print(robot_soul.explain())

    # Check Resonance
    resonance = elysia_soul.resonate_with(robot_soul)
    print(f"\nResonance between Elysia and LogicBot: {resonance:.4f}")

if __name__ == "__main__":
    test_sculptor()
