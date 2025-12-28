"""고양이 vs 사자 비교 테스트"""
import sys
sys.path.insert(0, "c:\\Elysia")

from Core._01_Foundation._02_Logic.multimodal_concept_node import get_multimodal_integrator

integrator = get_multimodal_integrator()

# 개념 생성
print("\n--- Building Concepts ---")
cat = integrator.build_concept_from_text(
    "고양이", 
    "고양이는 작고 부드러운 털을 가진 동물로 야옹 소리를 낸다"
)
lion = integrator.build_concept_from_text(
    "사자", 
    "사자는 크고 노란 털을 가진 동물로 으르렁 소리를 낸다"
)

# 비교
print("\n" + "="*50)
print("🐱 vs 🦁 COMPARISON")
print("="*50)

comparison = cat.compare_with(lion)

print(f"Overall Resonance: {comparison['overall_resonance']:.2f}")
print(f"Same Category: {comparison['is_same_category']}")
print(f"Distinct: {comparison['is_distinct']}")
print(f"\nShared modalities: {comparison['shared']}")
print(f"Different modalities: {comparison['different']}")

# 자기 수정 테스트
print("\n" + "="*50)
print("🔧 SELF-CORRECTION TEST")
print("="*50)

print(f"Before: {cat.modalities.get('texture', {})}")
cat.update_modality("texture", "부드럽고 포근한", 400.0)
print(f"After: {cat.modalities.get('texture', {})}")
print(f"Change history: {cat.change_history}")
