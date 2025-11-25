import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PREF_PATH = ROOT / "data" / "preferences.json"
STATE_DIR = ROOT / "data" / "background"
STATE_FILE = STATE_DIR / "state.json"
STOP_FILE = STATE_DIR / "stop.flag"
PID_FILE = STATE_DIR / "daemon.pid"


def load_prefs():
    try:
        return json.loads(PREF_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(st: dict):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_state():
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"last_keywords_ts": 0.0, "last_report_date": ""}


def run_module(mod: str, *args):
    try:
        cmd = [sys.executable, "-m", mod, *args]
        subprocess.run(cmd, cwd=str(ROOT), check=False)
    except Exception:
        pass


def daily_report_if_needed(state: dict):
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("last_report_date") != today:
        run_module("scripts.run_daily_report")
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [daily_report] generated for {today}")
        except Exception:
            pass
        state["last_report_date"] = today


def micro_sprint(state: dict):
    from tools import activity_registry as act
    try:
        act.start('background_learning', {'kind': 'micro_sprint'})
    except Exception:
        pass
    # Ingest newly added literature/journal files (cheap)
    run_module("scripts.ingest_literature", "--root", "data/corpus/literature")
    # Link keywords to concepts (cheap tf-idf) without viruses to keep low load
    try:
        from scripts.growth_sprint import collect_docs, build_tfidf, link_keywords_to_concepts
        from tools.kg_manager import KGManager
        kg = KGManager()
        docs = collect_docs(kg)
        if not docs:
            return
        tf, df = build_tfidf(docs)
        link_keywords_to_concepts(kg, docs, tf, df, topk=3)
    except Exception:
        pass
    finally:
        try:
            act.stop('background_learning')
        except Exception:
            pass


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        PID_FILE.write_text(str(os.getpid()), encoding='utf-8')
    except Exception:
        pass
    state = load_state()
    print("[background] Elysia micro-learner started. Ctrl+C to exit.")
    while True:
        if STOP_FILE.exists():
            try:
                STOP_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            print("[background] Stop flag detected. Exiting.")
            break
        prefs = load_prefs() or {}
        # Honor scheduled rest: background_resume_ts (epoch seconds)
        now = time.time()
        resume_ts = float(prefs.get("background_resume_ts", 0) or 0)
        if resume_ts and now >= resume_ts:
            prefs.pop("background_resume_ts", None)
            prefs["background_enabled"] = True
            try:
                PREF_PATH.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        enabled = bool(prefs.get("background_enabled", True)) and (now >= float(prefs.get("background_resume_ts", 0) or 0))
        interval = int(prefs.get("background_interval_sec", 900))  # default 15 min
        if enabled:
            t0 = time.time()
            try:
                try:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [background] micro_sprint start")
                except Exception:
                    pass
                micro_sprint(state)
                daily_report_if_needed(state)
                save_state(state)
            except Exception:
                pass
            # Sleep remaining interval from start time
            dt = time.time() - t0
            nap = max(30, interval - int(dt))
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [background] micro_sprint done in {int(dt)}s (next in {nap}s)")
            except Exception:
                pass
        else:
            nap = 60
        time.sleep(nap)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[background] Stopped by user.")

