"""
Neural Registry 테스트
======================
elysia_core 패키지의 기본 기능을 테스트합니다.
"""

import sys
sys.path.insert(0, "c:/Elysia")

from elysia_core import Cell, Organ

# 1. 테스트용 Cell 정의
@Cell("TestMemory")
class TestHippocampus:
    def remember(self, data):
        return f"Remembered: {data}"

@Cell("TestEmotion")
class TestAmygdala:
    def feel(self, emotion):
        return f"Feeling: {emotion}"

# 2. Organ을 통한 연결 테스트
def test_organ():
    print("\n🧪 Neural Registry Test")
    print("=" * 40)
    
    # Cell 목록 확인
    cells = Organ.list_cells()
    print(f"\n📋 Registered Cells: {cells}")
    
    # 메모리 Cell 가져오기
    memory = Organ.get("TestMemory")
    result = memory.remember("Hello, Organic World!")
    print(f"\n🧠 Memory: {result}")
    
    # 감정 Cell 가져오기
    emotion = Organ.get("TestEmotion")
    result = emotion.feel("Joy")
    print(f"❤️ Emotion: {result}")
    
    # 존재하지 않는 Cell 테스트
    print("\n🔍 Testing non-existent Cell...")
    try:
        Organ.get("NonExistent")
    except Exception as e:
        print(f"✅ Expected error: {type(e).__name__}")
    
    print("\n" + "=" * 40)
    print("✅ Neural Registry Test Passed!")

if __name__ == "__main__":
    test_organ()
