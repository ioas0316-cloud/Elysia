import sys
import mmap
import os
import struct
import math
import hashlib

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def string_to_phase(text):
    """문자열을 -pi ~ pi 사이의 위상각으로 해싱 (Semantic Entropy)"""
    hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    # 0.0 ~ 1.0
    normalized = (hash_val % 10000) / 10000.0
    # -pi ~ pi
    phase = (normalized * 2.0 - 1.0) * math.pi
    return phase

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inject_cognitive_shock.py <Thought or Code Change Summary>")
        sys.exit(1)

    thought_text = " ".join(sys.argv[1:])
    phase = string_to_phase(thought_text)
    
    FILENAME = "c:/Elysia/data/shared_manifold.bin"
    if not os.path.exists(FILENAME):
        print(f"[!] {FILENAME} not found. Engine must be running.")
        sys.exit(1)
        
    try:
        with open(FILENAME, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            
            # z축(Offset 24)에 인지적 텐션(Phase) 직접 타격 (Induction)
            data = struct.pack('d', phase)
            mm.seek(24)
            mm.write(data)
            
            mm.close()
            
        print(f"🌊 [우로보로스 피드백] '{thought_text}'")
        print(f"   -> 텍스트가 의미적 엔트로피(Phase: {phase:+.4f})로 강제 붕괴되어 엔진에 인가되었습니다.")
    except Exception as e:
        print(f"[!] Failed to inject phase: {e}")
