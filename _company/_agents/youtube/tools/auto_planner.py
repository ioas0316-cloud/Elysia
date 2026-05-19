#!/usr/bin/env python3
"""Auto Planner — runs trend_sniper.py on a fixed interval for a chosen
duration (e.g. overnight). Reads its config from auto_planner.json."""
import os, json, time, datetime, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "auto_planner.json")
SNIPER_PATH = os.path.join(HERE, "trend_sniper.py")

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일을 읽을 수 없어요: {CONFIG_PATH}\n{e}")
        sys.exit(1)

def main():
    cfg = load_config()
    interval_h = float(cfg.get("INTERVAL_HOURS", 6))  # v2.89.71: 디폴트 6시간 (하루 4번)
    total_h = float(cfg.get("TOTAL_RUN_HOURS", 0))    # v2.89.71: 0 = 무한 (24시간 자율 모드)

    # v2.89.71 — 24시간 자율 모드 본격 지원. TOTAL_RUN_HOURS=0이면 사용자가 멈출 때까지 무한.
    if total_h <= 0:
        print(f"\n🌙 [오토 플래너] 24시간 자율 모드 — {interval_h}시간마다 무한 반복")
        print(f"⚠️  사용자가 중단(Ctrl+C)할 때까지 계속 실행됩니다.")
        print(f"     백그라운드로 돌리려면 터미널에서:")
        print(f"     nohup python3 {os.path.abspath(__file__)} > planner.log 2>&1 &")
    else:
        print(f"\n🚀 [오토 플래너] {total_h}시간 동안 {interval_h}시간마다 트렌드 분석 (제한 모드)")
        print(f"⚠️  종료까지 {total_h}시간 채팅창 점유. Ctrl+C로 중단 가능.")
    print()

    if not os.path.exists(SNIPER_PATH):
        print(f"❌ trend_sniper.py를 찾을 수 없어요: {SNIPER_PATH}")
        sys.exit(1)
    # 첫 실행 전 trend_sniper.py가 정상 동작하는지 빠르게 검증
    print("🔍 trend_sniper.py 첫 회차 검증 중 (~30초)...")
    test_proc = subprocess.run([sys.executable, SNIPER_PATH], capture_output=True, text=True, timeout=300)
    if test_proc.returncode != 0:
        print(f"❌ trend_sniper.py 검증 실패 (exit {test_proc.returncode})")
        print("   먼저 trend_sniper.py 단독으로 ▶ 실행해서 설정·키워드·LLM 연결 확인 후 재시도.")
        if test_proc.stderr.strip():
            print("   에러 일부:")
            for line in test_proc.stderr.splitlines()[-5:]:
                print(f"   {line}")
        sys.exit(1)
    print("✅ 검증 완료. 본 루프 시작.\n")
    start = time.time()
    loop = 0
    while True:
        # v2.89.71 — total_h = 0이면 무한 (24시간 자율 모드)
        if total_h > 0 and (time.time() - start > total_h * 3600):
            print("\n☀️ 목표 가동 시간을 채웠어요. 종료합니다.")
            break
        loop += 1
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_h = (time.time() - start) / 3600
        print(f"\n[{ts}] 🤖 {loop}회차 트렌드 스나이핑 (가동 {elapsed_h:.1f}시간)")
        try:
            subprocess.run([sys.executable, SNIPER_PATH], check=False)
        except Exception as e:
            print(f"❌ 실행 실패: {e}")
        next_at = datetime.datetime.now() + datetime.timedelta(hours=interval_h)
        print(f"⏳ 다음 실행: {next_at.strftime('%Y-%m-%d %H:%M')} ({interval_h}시간 후)")
        time.sleep(interval_h * 3600)

if __name__ == "__main__":
    main()
