import argparse
from Project_Sophia.wisdom_virus import WisdomVirus, VirusEngine
from tools.kg_manager import KGManager


def main():
    parser = argparse.ArgumentParser(description="Run a Wisdom-Virus propagation demo")
    parser.add_argument("--statement", default="진정한 용기는 두려움이 없는 것이 아니라, 두려움에도 불구하고 나아가는 것이다.")
    parser.add_argument("--seeds", nargs="*", default=["concept:courage", "concept:fear", "concept:action"]) 
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()

    kg = KGManager()

    def mutate(host: str, text: str) -> str:
        if "role:soldier" in host:
            return "두려움에도 불구하고 전장으로 나아가는 용기"
        if "role:parent" in host:
            return "자녀를 지키기 위해 비난을 감수하는 용기"
        return text

    virus = WisdomVirus(
        id="wisdom:courage_redefined",
        statement=args.statement,
        seed_hosts=args.seeds,
        triggers=["용기", "두려움", "나아가"],
        mutate=mutate,
        max_hops=args.hops,
    )

    engine = VirusEngine(kg_manager=kg)
    engine.propagate(virus, context_tag="demo")
    print("[VirusDemo] Propagation completed.")


if __name__ == "__main__":
    main()

