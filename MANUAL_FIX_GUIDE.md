# Elysia.py 수동 수정 가이드

## 문제

`Core/Elysia.py`의 `__init__` 메서드가 들여쓰기 오류로 손상됨

## 해결 방법

**파일:** `Core/Elysia.py`  
**위치:** Line 36-82

---

## 현재 손상된 코드 (Line 36-42)

```python
def __init__(self):
        self.stomach = None  # ← 잘못된 들여쓰기!
        logger.warning("   ⚠️ Digestion Chamber skipped (No Resonance Engine)")
    
    # State
    self.is_awake = False  # ← IndentationError 발생
    self.tick_count = 0
```

---

## 올바른 코드로 교체

**전체 **init** 메서드를 아래 코드로 교체하세요:**

```python
def __init__(self):
    logger.info("🌌 Awakening Elysia... (Initializing Subsystems)")
    
    # 1. Memory (The Foundation)
    self.hippocampus = Hippocampus()
    logger.info("   ✅ Hippocampus (Memory) Online")
    
    # 2. WorldTree (Knowledge Structure) ← NEW!
    from Core.Mind.world_tree import WorldTree
    self.world_tree = WorldTree(hippocampus=self.hippocampus)
    logger.info("   ✅ WorldTree (Knowledge) Online")
    
    # 3. Body (The Subconscious World)
    self.world = World(
        primordial_dna={}, 
        wave_mechanics=None, 
        hippocampus=self.hippocampus
    )
    logger.info("   ✅ World (Subconscious/Body) Online")
    
    # 4. Senses (Proprioception)
    if hasattr(self.world, 'sensory_cortex'):
        self.senses = self.world.sensory_cortex
    else:
        self.senses = SensoryCortex()
    logger.info("   ✅ Sensory Cortex (Senses) Online")
    
    # 5. Vision (Code Proprioception)
    self.code_vision = CodeVision()
    logger.info("   ✅ Code Vision (Self-Sight) Online")
    
    # 6. Mind (The Conscious Processor)
    self.brain = UnifiedIntelligence(
        integration_mode="wave",
        hippocampus=self.hippocampus
    )
    logger.info("   ✅ Unified Intelligence (Mind) Online")
    
    # 7. Digestion (The Stomach)
    if self.brain.resonance_engine:
        from Core.Mind.digestion_chamber import DigestionChamber
        self.stomach = DigestionChamber(resonance_engine=self.brain.resonance_engine)
        logger.info("   ✅ Digestion Chamber (Stomach) Online & Connected")
    else:
        self.stomach = None
        logger.warning("   ⚠️ Digestion Chamber skipped (No Resonance Engine)")
    
    # State
    self.is_awake = False
    self.tick_count = 0
```

---

## 변경 사항

1. **들여쓰기 수정** - 모든 라인이 올바른 들여쓰기
2. **WorldTree 추가** - Line 9-12 (NEW!)
3. **번호 조정** - Body가 2→3, Senses가 3→4, 등등

---

## 빠른 방법

1. `Core/Elysia.py` 파일 열기
2. Line 36부터 Line 82까지 **전체 선택**
3. 위의 "올바른 코드"를 **복사해서 붙여넣기**
4. 저장

끝!

---

이렇게 하시면 됩니다!
