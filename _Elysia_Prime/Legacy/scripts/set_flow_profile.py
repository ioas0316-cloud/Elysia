# [Genesis: 2025-12-02] Purified by Elysia
import argparse
import os

PROFILE_FILE = os.path.join("data", "flows", "profile.txt")


def main():
    ap = argparse.ArgumentParser(description="Set FlowEngine profile (generic/learning or path to yaml)")
    ap.add_argument("--profile", required=True, help="'generic' or 'learning' or a yaml path")
    args = ap.parse_args()
    prof = args.profile.strip()
    if prof in ("generic", "learning"):
        # write short name; FlowEngine resolves to data/flows/{name}.yaml
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            f.write(prof)
        print(f"[set_flow_profile] Set profile to '{prof}'")
    else:
        # treat as explicit path
        if not os.path.exists(prof):
            raise SystemExit(f"Not found: {prof}")
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            f.write(os.path.abspath(prof))
        print(f"[set_flow_profile] Set profile to '{prof}'")


if __name__ == "__main__":
    main()
