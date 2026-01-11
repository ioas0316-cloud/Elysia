"""
Run Elysia's Guardian lifecycle (AWAKE/IDLE + dream cycle) in a console.
"""
from Project_Elysia.guardian import Guardian


def main():
    g = Guardian()
    g.monitor_and_protect()


if __name__ == "__main__":
    main()

