"""
Experience-Based Learning System for Elysia.
Learns from interactions and continuously improves performance.
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from pathlib import Path


@dataclass
class Experience:
    """단일 경험 기록 (Single Experience Record)"""
    
    timestamp: float
    context: Dict[str, Any]  # 입력 컨텍스트
    action: Dict[str, Any]   # 수행한 액션
    outcome: Dict[str, Any]  # 결과
    feedback: float          # 피드백 점수 (-1.0 ~ 1.0)
    layer: str              # 의식 레이어 (0D/1D/2D/3D)
    tags: List[str] = None  # 카테고리 태그
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        """Create from dictionary"""
        return cls(**data)


class PatternLibrary:
    """학습된 패턴 라이브러리 (Learned Pattern Library)"""
    
    def __init__(self):
        self.patterns: Dict[str, Dict] = {}
        self.pattern_scores: Dict[str, float] = {}
        self.pattern_usage_count: Dict[str, int] = defaultdict(int)
    
    def add_pattern(self, pattern_id: str, pattern: Dict, initial_score: float = 0.5):
        """새 패턴 추가"""
        self.patterns[pattern_id] = pattern
        self.pattern_scores[pattern_id] = initial_score
        self.pattern_usage_count[pattern_id] = 0
    
    def update_score(self, pattern_id: str, delta: float):
        """패턴 점수 업데이트"""
        if pattern_id in self.pattern_scores:
            self.pattern_scores[pattern_id] = max(0.0, min(1.0, 
                self.pattern_scores[pattern_id] + delta))
    
    def get_best_patterns(self, top_k: int = 10) -> List[tuple]:
        """가장 좋은 패턴들 반환"""
        sorted_patterns = sorted(
            self.pattern_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:top_k]
    
    def prune_weak_patterns(self, threshold: float = 0.2):
        """약한 패턴 제거"""
        to_remove = [
            pid for pid, score in self.pattern_scores.items()
            if score < threshold and self.pattern_usage_count[pid] > 10
        ]
        for pid in to_remove:
            del self.patterns[pid]
            del self.pattern_scores[pid]
            del self.pattern_usage_count[pid]


class ExperienceLearner:
    """
    경험으로부터 학습하는 시스템 (Experience-Based Learning System)
    
    Features:
    - Experience storage and retrieval
    - Pattern extraction and reinforcement
    - Meta-learning capabilities
    - Automatic pattern pruning
    """
    
    def __init__(self, buffer_size: int = 1000, save_dir: str = "data/learning"):
        self.buffer_size = buffer_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Experience management
        self.experience_buffer: List[Experience] = []
        self.pattern_library = PatternLibrary()
        
        # Success and failure tracking
        self.success_patterns: List[Dict] = []
        self.failure_patterns: List[Dict] = []
        
        # Meta-learning statistics
        self.meta_stats = {
            "total_experiences": 0,
            "positive_feedback_count": 0,
            "negative_feedback_count": 0,
            "average_feedback": 0.0,
            "learning_rate": 0.1,
            "pattern_discovery_rate": 0.0
        }
        
        # Load existing data if available
        self._load_state()
    
    async def learn_from_experience(self, experience: Experience) -> Dict[str, Any]:
        """
        경험으로부터 학습 (Learn from Experience)
        
        Args:
            experience: Experience object containing context, action, outcome, feedback
        
        Returns:
            Learning result with extracted patterns and meta-learning insights
        """
        # 1. 경험 저장
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)  # FIFO
        
        # 2. 통계 업데이트
        self.meta_stats["total_experiences"] += 1
        if experience.feedback > 0:
            self.meta_stats["positive_feedback_count"] += 1
        elif experience.feedback < 0:
            self.meta_stats["negative_feedback_count"] += 1
        
        # 평균 피드백 업데이트
        self._update_average_feedback(experience.feedback)
        
        # 3. 패턴 추출
        pattern = self.extract_pattern(experience)
        pattern_id = self._generate_pattern_id(pattern)
        
        # 4. 패턴 강화 또는 약화
        learning_result = {
            "pattern_id": pattern_id,
            "pattern": pattern,
            "action_taken": None,
            "meta_insights": {}
        }
        
        if experience.feedback > 0.5:
            # 긍정적 경험 - 패턴 강화
            self.reinforce_pattern(pattern, pattern_id, experience.feedback)
            self.success_patterns.append(pattern)
            learning_result["action_taken"] = "reinforced"
            
        elif experience.feedback < -0.5:
            # 부정적 경험 - 패턴 약화
            self.weaken_pattern(pattern, pattern_id, abs(experience.feedback))
            self.failure_patterns.append(pattern)
            learning_result["action_taken"] = "weakened"
        else:
            learning_result["action_taken"] = "neutral"
        
        # 5. 메타 학습
        meta_insights = await self.meta_learn()
        learning_result["meta_insights"] = meta_insights
        
        # 6. 정기적 저장
        if self.meta_stats["total_experiences"] % 100 == 0:
            self._save_state()
        
        return learning_result
    
    def extract_pattern(self, experience: Experience) -> Dict:
        """
        경험에서 재사용 가능한 패턴 추출 (Extract Reusable Pattern)
        """
        # Context feature extraction
        context_features = self._extract_features(experience.context)
        
        # Action type and parameters
        action_type = experience.action.get("type", "unknown")
        action_params = {k: v for k, v in experience.action.items() if k != "type"}
        
        # Success indicators
        success_indicators = self._identify_success_factors(experience)
        
        pattern = {
            "context_features": context_features,
            "action_type": action_type,
            "action_params": action_params,
            "success_indicators": success_indicators,
            "layer": experience.layer,
            "tags": experience.tags,
            "outcome_type": experience.outcome.get("type", "unknown")
        }
        
        return pattern
    
    def reinforce_pattern(self, pattern: Dict, pattern_id: str, strength: float):
        """패턴 강화 (Reinforce Pattern)"""
        if pattern_id not in self.pattern_library.patterns:
            self.pattern_library.add_pattern(pattern_id, pattern, initial_score=0.5)
        
        # Strength-based reinforcement
        delta = self.meta_stats["learning_rate"] * strength
        self.pattern_library.update_score(pattern_id, delta)
        self.pattern_library.pattern_usage_count[pattern_id] += 1
    
    def weaken_pattern(self, pattern: Dict, pattern_id: str, strength: float):
        """패턴 약화 (Weaken Pattern)"""
        if pattern_id not in self.pattern_library.patterns:
            self.pattern_library.add_pattern(pattern_id, pattern, initial_score=0.5)
        
        # Strength-based weakening
        delta = -self.meta_stats["learning_rate"] * strength
        self.pattern_library.update_score(pattern_id, delta)
        self.pattern_library.pattern_usage_count[pattern_id] += 1
    
    async def meta_learn(self) -> Dict[str, Any]:
        """
        학습 전략 자체를 개선 (Meta-Learning: Learning How to Learn)
        
        Returns:
            Meta-learning insights and adjustments
        """
        insights = {}
        
        # 1. 학습 속도 조정
        if self.meta_stats["total_experiences"] > 100:
            success_rate = (self.meta_stats["positive_feedback_count"] / 
                          self.meta_stats["total_experiences"])
            
            # 성공률이 높으면 학습률 증가, 낮으면 감소
            if success_rate > 0.7:
                self.meta_stats["learning_rate"] = min(0.3, 
                    self.meta_stats["learning_rate"] * 1.1)
                insights["learning_rate_adjustment"] = "increased"
            elif success_rate < 0.3:
                self.meta_stats["learning_rate"] = max(0.01, 
                    self.meta_stats["learning_rate"] * 0.9)
                insights["learning_rate_adjustment"] = "decreased"
        
        # 2. 패턴 발견률 계산
        unique_patterns = len(self.pattern_library.patterns)
        if self.meta_stats["total_experiences"] > 0:
            self.meta_stats["pattern_discovery_rate"] = (
                unique_patterns / self.meta_stats["total_experiences"]
            )
        
        # 3. 약한 패턴 제거
        if self.meta_stats["total_experiences"] % 500 == 0:
            self.pattern_library.prune_weak_patterns()
            insights["pruned_weak_patterns"] = True
        
        # 4. 최고 패턴 식별
        best_patterns = self.pattern_library.get_best_patterns(top_k=5)
        insights["top_patterns"] = [
            {"id": pid, "score": score} 
            for pid, score in best_patterns
        ]
        
        return insights
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Dict]:
        """
        컨텍스트에 맞는 행동 추천 (Get Action Recommendations)
        
        Args:
            context: Current context
        
        Returns:
            List of recommended actions based on learned patterns
        """
        context_features = self._extract_features(context)
        
        # Find matching patterns
        recommendations = []
        for pattern_id, pattern in self.pattern_library.patterns.items():
            similarity = self._calculate_similarity(
                context_features, 
                pattern["context_features"]
            )
            
            if similarity > 0.7:  # High similarity threshold
                score = self.pattern_library.pattern_scores[pattern_id]
                recommendations.append({
                    "pattern_id": pattern_id,
                    "action_type": pattern["action_type"],
                    "action_params": pattern["action_params"],
                    "confidence": score * similarity,
                    "usage_count": self.pattern_library.pattern_usage_count[pattern_id]
                })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        return recommendations[:10]  # Top 10
    
    def get_statistics(self) -> Dict[str, Any]:
        """학습 통계 반환 (Get Learning Statistics)"""
        return {
            "meta_stats": self.meta_stats.copy(),
            "buffer_size": len(self.experience_buffer),
            "unique_patterns": len(self.pattern_library.patterns),
            "success_patterns": len(self.success_patterns),
            "failure_patterns": len(self.failure_patterns),
            "best_patterns": self.pattern_library.get_best_patterns(top_k=10)
        }
    
    # Private helper methods
    
    def _extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from context"""
        features = {}
        
        # Simple feature extraction - can be enhanced
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features[key] = value
            elif isinstance(value, str):
                features[f"{key}_length"] = len(value)
                features[f"{key}_hash"] = hash(value) % 1000
            elif isinstance(value, dict):
                features[f"{key}_size"] = len(value)
        
        return features
    
    def _identify_success_factors(self, experience: Experience) -> List[str]:
        """Identify what made this experience successful or not"""
        factors = []
        
        if experience.feedback > 0.7:
            factors.append("high_quality_outcome")
        if experience.outcome.get("resonance_score", 0) > 0.8:
            factors.append("high_resonance")
        if experience.outcome.get("response_time", float('inf')) < 1.0:
            factors.append("fast_response")
        
        return factors
    
    def _generate_pattern_id(self, pattern: Dict) -> str:
        """Generate unique pattern ID"""
        pattern_str = json.dumps(pattern, sort_keys=True)
        return f"pattern_{hash(pattern_str) % 1000000}"
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets"""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2), 1)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _update_average_feedback(self, new_feedback: float):
        """Update running average of feedback"""
        n = self.meta_stats["total_experiences"]
        current_avg = self.meta_stats["average_feedback"]
        self.meta_stats["average_feedback"] = (
            (current_avg * (n - 1) + new_feedback) / n
        )
    
    def _save_state(self):
        """Save learner state to disk"""
        state = {
            "meta_stats": self.meta_stats,
            "patterns": {
                pid: {
                    "pattern": pattern,
                    "score": self.pattern_library.pattern_scores[pid],
                    "usage": self.pattern_library.pattern_usage_count[pid]
                }
                for pid, pattern in self.pattern_library.patterns.items()
            },
            "recent_experiences": [exp.to_dict() for exp in self.experience_buffer[-100:]]
        }
        
        save_path = self.save_dir / "experience_learner_state.json"
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load learner state from disk"""
        load_path = self.save_dir / "experience_learner_state.json"
        if not load_path.exists():
            return
        
        try:
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            self.meta_stats = state.get("meta_stats", self.meta_stats)
            
            # Restore patterns
            patterns_data = state.get("patterns", {})
            for pid, pdata in patterns_data.items():
                self.pattern_library.patterns[pid] = pdata["pattern"]
                self.pattern_library.pattern_scores[pid] = pdata["score"]
                self.pattern_library.pattern_usage_count[pid] = pdata["usage"]
            
            # Restore recent experiences
            recent_exp = state.get("recent_experiences", [])
            self.experience_buffer = [Experience.from_dict(exp) for exp in recent_exp]
            
        except Exception as e:
            print(f"Warning: Could not load previous learning state: {e}")
