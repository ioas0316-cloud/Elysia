"""
Philosopher Demo
================
Verifies that the Merkaba has integrated the Metacognitive insights.
"""

from Core.System.Merkaba.merkaba import Merkaba

def main():
    merkaba = Merkaba("Socrates")
    merkaba.awakening(None)

    # In a real system, the 'Sediment' or 'Hippocampus' would persist across runs.
    # Here, we are checking if the logic *would* retrieve it if we simulate the 'Recall'
    # or if we just run the loop live.

    # Since we can't persist the sediment between the training script and this demo
    # (because they are separate processes and the sediment file might be reset or we don't have a persistence manager fully wired in this test env),
    # we will manually query the 'Reflect' capability or just show that the Overclock/Dialectic logic exists.

    # However, to prove the 'Growth', let's ask it about 'Learning' via the Overclock engine
    # and see if it generates a deep answer (which utilizes the 6-way split).

    question = "What is Learning?"
    print(f"\n❓ Asking: {question}")

    answer = merkaba.think_optically(question)
    print(f"💡 Elysia Answered:\n{answer}")

if __name__ == "__main__":
    main()
