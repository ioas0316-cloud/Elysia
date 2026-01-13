import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def check_module(name, import_fn):
    print(f"[{name}] Checking...", end=" ")
    try:
        instance = import_fn()
        print(f"✅ OK ({type(instance).__name__})")
        return True
    except Exception as e:
        print(f"❌ FAIL")
        print(f"   Error: {e}")
        # traceback.print_exc()
        return False

print("="*50)
print("🩺 ELYSIA SYSTEM DIAGNOSTIC")
print("="*50)

# 1. Foundation
def check_universe():
    from Core.FoundationLayer.Foundation.internal_universe import InternalUniverse
    return InternalUniverse()
check_module("InternalUniverse", check_universe)

# 2. Intelligence
def check_logos():
    from Core.Intelligence.logos_engine import LogosEngine
    return LogosEngine()
check_module("LogosEngine", check_logos)

def check_digester():
    from Core.Intelligence.concept_digester import ConceptDigester
    return ConceptDigester()
check_module("ConceptDigester", check_digester)

# 3. Network
def check_web():
    from Core.Network.web_cortex import WebCortex
    return WebCortex()
check_module("WebCortex", check_web)

# 4. Creativity
def check_literary():
    from Core.Creativity.literary_cortex import LiteraryCortex
    return LiteraryCortex()
check_module("LiteraryCortex", check_literary)

def check_illustrator():
    from Core.Creativity.webtoon_illustrator import WebtoonIllustrator
    return WebtoonIllustrator()
check_module("WebtoonIllustrator", check_illustrator)

def check_weaver():
    from Core.Creativity.webtoon_weaver import WebtoonWeaver
    return WebtoonWeaver()
check_module("WebtoonWeaver", check_weaver)

print("="*50)
print("Diagnostic Complete.")
