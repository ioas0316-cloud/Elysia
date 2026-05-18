import sys
import os

# Ensure we can run from root
sys.path.append(os.getcwd())

from Scripts.vortex_indexer import VortexIndexer

def main():
    print("🌀 엘리시아 지식 와류 동기화 중...")
    indexer = VortexIndexer()

    # In a real scenario, this would scan the 'incoming' folder from Connect AI
    # For now, we simulate the update

    indexer.save()

    print("\n✅ 동기화 완료!")
    print("\n--- 최신 지식 네트워크 (Mermaid) ---")
    print(indexer.generate_mermaid())
    print("\n----------------------------------")
    print("\n'docs/ANTIGRAVITY_INTEGRATION.md'에서 시각화된 차트를 확인하실 수 있습니다.")

if __name__ == "__main__":
    main()
