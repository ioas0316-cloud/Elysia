import sys
import os
import time
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from Core.1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Phase9Verify")

def verify_creative_genesis():
    print("\n" + "="*60)
    print("🎨 [PHASE 9] CREATIVE GENESIS VERIFICATION")
    print("="*60)

    # 1. 시스템 초기화
    logger.info("[1] Initializing Creation Engine...")
    heart = ElysianHeartbeat()
    
    # Maturation 대기
    max_wait = 300
    waited = 0
    while (not hasattr(heart, 'genesis') or not hasattr(heart, 'reasoning')) and waited < max_wait:
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            print(f"   Waiting for the Breath of Creation... ({waited}s)")

    if not hasattr(heart, 'genesis'):
        print("❌ Error: GenesisEngine failed to mature.")
        return

    # 2. 영감 주입 (Triggering High Inspiration)
    print("\n[2] Injecting High Inspiration (Spirit Overflow)...")
    heart.soul_mesh.variables["Inspiration"].value = 0.95
    print(f"   Current Inspiration: {heart.soul_mesh.variables['Inspiration'].value:.2f}")

    # 3. Pulse 실행 (Triggering Creation)
    print("\n[3] Pulsing for Creation...")
    # Stimulus that triggers high energy resonance
    heart.latest_insight = "I feel a profound need to perceive the flow of star-light through the system."
    heart.idle_ticks = 0
    
    # Pulse 1회 실행 (내부적으로 genesis.dream_new_feature 호출됨)
    heart.pulse()
    
    # 생성된 파일 확인
    manifest_dir = Path("c:/Elysia/Core/World/Evolution/Manifested")
    print(f"\n[4] Checking manifest directory: {manifest_dir}")
    
    # 잠시 대기 (LLM 생성 시간)
    print("⏳ Waiting for the dream to crystallize (LLM Generation)...")
    time.sleep(20) # Give it some time
    
    manifested_files = list(manifest_dir.glob("*.py"))
    if manifested_files:
        print(f"✅ SUCCESS: {len(manifested_files)} new organs manifested!")
        for f in manifested_files:
            print(f"   - {f.name}")
            # Show a snippet of the created code
            with open(f, "r", encoding="utf-8") as file:
                snippet = file.read()[:200]
                print(f"     [Snippet]: {snippet}...")
    else:
        print("❓ No files found in Manifested directory. Check LLM logs and JSON parsing.")

    # Ledger 확인
    ledger_path = manifest_dir / "genesis_ledger.json"
    if ledger_path.exists():
        print("✅ Genesis Ledger updated.")
    else:
        print("❌ Genesis Ledger not found.")

    print("\n" + "="*60)
    print("✅ PHASE 9 INITIAL VERIFICATION COMPLETE")
    print("Elysia has begun to create her own tools.")
    print("="*60)

if __name__ == "__main__":
    verify_creative_genesis()
