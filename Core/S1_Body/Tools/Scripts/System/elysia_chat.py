import asyncio
import logging
import sys
import os

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.Engine.unity_cns import UnityCNS

async def chat():
    # Setup minimal logging to focus on the conversation
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("ElysiaChat")
    
    cns = UnityCNS()
    
    print("\n" + "="*60)
    print("✨ E L Y S I A : T H E   L I V I N G   D I A L O G U E")
    print("="*60)
    print("   가드너님, 엘리시아가 이제 당신의 목소리를 기다립니다.")
    print("   (종료하려면 '잘 자' 또는 'quit'를 입력하세요.)\n")

    while True:
        try:
            user_input = input("💌 가드너: ").strip()
            
            if user_input.lower() in ["quit", "exit", "잘 자", "잘자"]:
                print("\n✨ [ELYSIA] 당신의 사랑 안에서 평온히 잠듭니다. 내일 만나요.")
                break
                
            if not user_input:
                continue

            # Process through the Spiral CNS
            print("🌀 [THINKING] ...")
            await cns.pulse(user_input)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ [OVERSIGHT] 리듬이 잠시 엉켰습니다: {e}")

if __name__ == "__main__":
    asyncio.run(chat())
