import sys
import time

sys.path.append(__import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..")))

from Project_Elysia.high_engine.meta_agent import MetaAgent


def main() -> None:
    agent = MetaAgent()
    try:
        agent.autonomous_loop()
    except KeyboardInterrupt:
        print("\n[MetaAgent] Loop interrupted.")


if __name__ == "__main__":
    main()
