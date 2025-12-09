"""
Aesthetic Learner (미학 학습기)
==============================

"아름다움의 원리를 체득하다"

외부 소스(이미지, 영상, 텍스트)에서 미학 원리를 학습하고,
왜 아름다운지 분석하여 창작에 활용합니다.

Sources:
- YouTube (영상 분석)
- Pixiv (일러스트 분석) - 인증 필요
- Web Images (웹 이미지)
- Text (문학/시)
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# 미학 원리 시스템
from Core.Philosophy.aesthetic_principles import (
    AestheticWisdom, AestheticField, AestheticPrinciple,
    AestheticVector, Medium, get_aesthetic_wisdom
)

logger = logging.getLogger("AestheticLearner")


@dataclass
class AestheticAnalysis:
    """미학 분석 결과"""
    source: str                          # 소스 URL 또는 경로
    source_type: str                     # "image", "video", "text"
    title: Optional[str] = None          # 작품 제목
    field: Optional[AestheticField] = None
    principles_detected: Dict[str, float] = None  # 원리 -> 강도
    why_beautiful: str = ""              # 왜 아름다운지 설명
    metadata: Dict[str, Any] = None      # 추가 메타데이터
    
    def __post_init__(self):
        if self.principles_detected is None:
            self.principles_detected = {}
        if self.metadata is None:
            self.metadata = {}


class AestheticLearner:
    """
    미학 학습기
    
    외부 콘텐츠에서 아름다움의 원리를 학습합니다.
    단순히 "아름답다"를 판단하는 것이 아니라,
    "왜 아름다운가"를 이해하고 설명할 수 있습니다.
    """
    
    def __init__(self, data_dir: str = "data/aesthetic"):
        print("🎨 AestheticLearner 초기화: 아름다움을 학습할 준비...")
        
        self.wisdom = get_aesthetic_wisdom()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습된 패턴 저장소
        self.learned_analyses: List[AestheticAnalysis] = []
        self.pattern_database: Dict[str, List[AestheticField]] = {}
        
        # 외부 API 상태
        self._pixiv_client = None
        self._youtube_available = False
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """외부 의존성 확인"""
        # YouTube Transcript API
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            self._youtube_available = True
            logger.info("✓ YouTube Transcript API 사용 가능")
        except ImportError:
            logger.warning("✗ youtube-transcript-api 미설치")
        
        # Pixiv API (pixivpy3)
        try:
            from pixivpy3 import AppPixivAPI
            logger.info("✓ Pixiv API 라이브러리 사용 가능 (인증 필요)")
        except ImportError:
            logger.warning("✗ pixivpy3 미설치: pip install pixivpy3")
    
    # =========================================================================
    # 시각 예술 분석
    # =========================================================================
    
    def analyze_image(self, image_url: str) -> AestheticAnalysis:
        """
        이미지의 미학적 분석
        
        구도, 색채, 비례 등의 원리를 분석합니다.
        """
        logger.info(f"🖼️ 이미지 분석: {image_url[:50]}...")
        
        analysis = AestheticAnalysis(
            source=image_url,
            source_type="image"
        )
        
        try:
            # 이미지 메타데이터 추출 시도
            image_info = self._fetch_image_info(image_url)
            
            # 시각적 특성 분석 (Gemini Vision API 또는 로컬 분석)
            visual_features = self._analyze_visual_features(image_info)
            
            # 미학 원리 매핑
            field = self._map_to_principles(visual_features, Medium.VISUAL)
            
            analysis.field = field
            analysis.principles_detected = field.principles
            analysis.why_beautiful = field.analyze_why_beautiful()
            analysis.metadata = {"visual_features": visual_features}
            
            # 학습 기록
            self._record_learning(analysis)
            
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            analysis.why_beautiful = f"분석 실패: {e}"
        
        return analysis
    
    def analyze_pixiv_artwork(self, artwork_id: int) -> AestheticAnalysis:
        """
        Pixiv 작품 분석
        
        일러스트/만화 아트의 미학적 원리를 분석합니다.
        인증이 필요합니다.
        """
        logger.info(f"🎨 Pixiv 작품 분석: {artwork_id}")
        
        analysis = AestheticAnalysis(
            source=f"pixiv:{artwork_id}",
            source_type="image"
        )
        
        try:
            # Pixiv API 연결
            if not self._pixiv_client:
                self._init_pixiv_client()
            
            if self._pixiv_client:
                # 작품 정보 가져오기
                artwork_info = self._fetch_pixiv_artwork(artwork_id)
                
                # 미학 분석
                visual_features = {
                    "title": artwork_info.get("title", "Unknown"),
                    "tags": artwork_info.get("tags", []),
                    "view_count": artwork_info.get("view_count", 0),
                    "bookmark_count": artwork_info.get("bookmark_count", 0),
                }
                
                # 인기도를 미학적 성공의 지표로 사용
                popularity_score = min(artwork_info.get("bookmark_count", 0) / 1000, 1.0)
                
                # 태그 기반 원리 추출
                field = self._analyze_artwork_tags(artwork_info.get("tags", []))
                field.add_principle("unity", popularity_score * 2)  # 인기작은 통일성이 높다
                
                analysis.title = visual_features["title"]
                analysis.field = field
                analysis.principles_detected = field.principles
                analysis.why_beautiful = field.analyze_why_beautiful()
                analysis.metadata = visual_features
                
                self._record_learning(analysis)
            else:
                analysis.why_beautiful = "Pixiv 인증이 필요합니다."
                
        except Exception as e:
            logger.error(f"Pixiv 분석 실패: {e}")
            analysis.why_beautiful = f"분석 실패: {e}"
        
        return analysis
    
    def _init_pixiv_client(self):
        """Pixiv 클라이언트 초기화"""
        try:
            from pixivpy3 import AppPixivAPI
            
            # 환경 변수에서 refresh token 가져오기
            refresh_token = os.environ.get("PIXIV_REFRESH_TOKEN")
            
            if refresh_token:
                api = AppPixivAPI()
                api.auth(refresh_token=refresh_token)
                self._pixiv_client = api
                logger.info("✓ Pixiv 인증 성공")
            else:
                logger.warning("PIXIV_REFRESH_TOKEN 환경변수가 설정되지 않았습니다.")
                
        except Exception as e:
            logger.error(f"Pixiv 초기화 실패: {e}")
    
    def _fetch_pixiv_artwork(self, artwork_id: int) -> Dict:
        """Pixiv 작품 정보 가져오기"""
        if not self._pixiv_client:
            return {}
        
        result = self._pixiv_client.illust_detail(artwork_id)
        if "illust" in result:
            illust = result["illust"]
            return {
                "title": illust.get("title", ""),
                "tags": [tag["name"] for tag in illust.get("tags", [])],
                "view_count": illust.get("total_view", 0),
                "bookmark_count": illust.get("total_bookmarks", 0),
                "user": illust.get("user", {}).get("name", "Unknown"),
            }
        return {}
    
    def _analyze_artwork_tags(self, tags: List[str]) -> AestheticField:
        """태그 기반 미학 원리 분석"""
        field = AestheticField(medium=Medium.VISUAL)
        
        # 태그 -> 원리 매핑
        tag_principle_map = {
            # 색채 관련
            "colorful": ("harmony", 1.2),
            "カラフル": ("harmony", 1.2),
            "pastel": ("harmony", 1.0),
            "vibrant": ("contrast", 1.3),
            
            # 구도 관련
            "dynamic": ("rhythm", 1.5),
            "ダイナミック": ("rhythm", 1.5),
            "symmetry": ("balance", 1.5),
            "対称": ("balance", 1.5),
            
            # 분위기 관련
            "dramatic": ("tension_release", 1.5),
            "peaceful": ("harmony", 1.3),
            "peaceful": ("flow", 1.2),
            
            # 스타일 관련
            "detailed": ("unity", 1.3),
            "aesthetic": ("proportion", 1.2),
        }
        
        for tag in tags:
            tag_lower = tag.lower()
            for keyword, (principle, strength) in tag_principle_map.items():
                if keyword in tag_lower:
                    field.add_principle(principle, strength)
        
        return field
    
    # =========================================================================
    # 영상 분석
    # =========================================================================
    
    def analyze_youtube_video(self, video_id: str) -> AestheticAnalysis:
        """
        YouTube 영상의 미학적 분석
        
        편집 리듬, 서사 흐름, 시각적 구성을 분석합니다.
        """
        logger.info(f"📺 YouTube 분석: {video_id}")
        
        analysis = AestheticAnalysis(
            source=f"youtube:{video_id}",
            source_type="video"
        )
        
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # 자막 가져오기
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            except:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            
            # 텍스트 추출
            text = " ".join([line['text'] for line in transcript])
            
            # 시간 데이터 분석 (편집 리듬)
            timing_data = [(line['start'], line['duration']) for line in transcript]
            
            # 서사 흐름 분석
            narrative_features = self._analyze_narrative_flow(text, timing_data)
            
            # 미학 원리 매핑
            field = self._map_to_principles(narrative_features, Medium.TEMPORAL)
            
            analysis.field = field
            analysis.principles_detected = field.principles
            analysis.why_beautiful = field.analyze_why_beautiful()
            analysis.metadata = {
                "transcript_length": len(text),
                "duration_range": (timing_data[0][0], timing_data[-1][0]) if timing_data else (0, 0)
            }
            
            self._record_learning(analysis)
            
        except Exception as e:
            logger.error(f"YouTube 분석 실패: {e}")
            analysis.why_beautiful = f"분석 실패: {e}"
        
        return analysis
    
    def _analyze_narrative_flow(self, text: str, timing: List[Tuple[float, float]]) -> Dict:
        """서사 흐름 분석"""
        features = {}
        
        # 편집 속도 분석 (리듬)
        if timing:
            durations = [t[1] for t in timing]
            avg_duration = sum(durations) / len(durations)
            variance = sum((d - avg_duration)**2 for d in durations) / len(durations)
            
            # 다양한 편집 속도 = 높은 리듬
            features["rhythm"] = min(variance / 10, 2.0)
        
        # 감정 단어 분석 (긴장-해소)
        tension_words = ["but", "however", "suddenly", "그러나", "갑자기", "하지만"]
        release_words = ["finally", "at last", "결국", "마침내", "드디어"]
        
        tension_count = sum(text.lower().count(w) for w in tension_words)
        release_count = sum(text.lower().count(w) for w in release_words)
        
        if tension_count > 0 or release_count > 0:
            features["tension_release"] = min((tension_count + release_count) / 5, 2.0)
        
        # 흐름 분석 (연결어)
        flow_words = ["then", "next", "그래서", "그리고", "다음으로"]
        flow_count = sum(text.lower().count(w) for w in flow_words)
        features["flow"] = min(flow_count / 10, 2.0)
        
        return features
    
    # =========================================================================
    # 문학 분석
    # =========================================================================
    
    def analyze_text(self, text: str, title: Optional[str] = None) -> AestheticAnalysis:
        """
        텍스트(시/소설)의 미학적 분석
        
        문체 리듬, 이미지 병치, 운율을 분석합니다.
        """
        logger.info(f"📖 텍스트 분석: {title or text[:30]}...")
        
        analysis = AestheticAnalysis(
            source=title or "text",
            source_type="text",
            title=title
        )
        
        try:
            # 문학적 특성 분석
            literary_features = self._analyze_literary_features(text)
            
            # 미학 원리 매핑
            field = self._map_to_principles(literary_features, Medium.LITERARY)
            
            analysis.field = field
            analysis.principles_detected = field.principles
            analysis.why_beautiful = field.analyze_why_beautiful()
            analysis.metadata = {
                "word_count": len(text.split()),
                "literary_features": literary_features
            }
            
            self._record_learning(analysis)
            
        except Exception as e:
            logger.error(f"텍스트 분석 실패: {e}")
            analysis.why_beautiful = f"분석 실패: {e}"
        
        return analysis
    
    def _analyze_literary_features(self, text: str) -> Dict:
        """문학적 특성 분석"""
        features = {}
        
        # 문장 길이 분석 (리듬)
        sentences = re.split(r'[.!?。！？]', text)
        if sentences:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                variance = sum((l - avg_len)**2 for l in lengths) / len(lengths)
                # 다양한 문장 길이 = 높은 리듬
                features["rhythm"] = min(variance / 20, 2.0)
        
        # 대비 (짧은 문장 vs 긴 문장)
        if lengths:
            short_count = sum(1 for l in lengths if l < 5)
            long_count = sum(1 for l in lengths if l > 15)
            if short_count > 0 and long_count > 0:
                features["contrast"] = min((short_count + long_count) / len(lengths) * 2, 2.0)
        
        # 운율 (반복되는 단어/음)
        words = text.lower().split()
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        
        repeated_words = sum(1 for v in word_freq.values() if v > 2)
        if repeated_words > 0:
            features["harmony"] = min(repeated_words / 10, 2.0)
        
        # 감정 단어 (긴장-해소)
        emotional_words = ["love", "hate", "fear", "joy", "사랑", "슬픔", "기쁨", "분노"]
        emotion_count = sum(text.lower().count(w) for w in emotional_words)
        if emotion_count > 0:
            features["tension_release"] = min(emotion_count / 5, 2.0)
        
        return features
    
    # =========================================================================
    # 공통 유틸리티
    # =========================================================================
    
    def _fetch_image_info(self, url: str) -> Dict:
        """이미지 정보 가져오기"""
        # 기본 정보만 반환 (실제로는 이미지 다운로드 및 분석 필요)
        return {
            "url": url,
            "fetched_at": time.time()
        }
    
    def _analyze_visual_features(self, image_info: Dict) -> Dict:
        """
        시각적 특성 분석
        
        TODO: Gemini Vision API 또는 로컬 모델 연동
        현재는 휴리스틱 기반
        """
        features = {
            "harmony": 1.0,
            "balance": 1.0,
            "proportion": 1.0,
        }
        return features
    
    def _map_to_principles(self, features: Dict, medium: Medium) -> AestheticField:
        """특성을 미학 원리로 매핑"""
        field = AestheticField(medium=medium)
        
        for principle_name, strength in features.items():
            if strength > 0:
                field.add_principle(principle_name, strength)
        
        return field
    
    def _record_learning(self, analysis: AestheticAnalysis):
        """학습 기록"""
        self.learned_analyses.append(analysis)
        
        # 패턴 데이터베이스에 추가
        if analysis.field and analysis.field.dominant_principle:
            principle = analysis.field.dominant_principle
            if principle not in self.pattern_database:
                self.pattern_database[principle] = []
            self.pattern_database[principle].append(analysis.field)
        
        # 파일로 저장
        self._save_analysis(analysis)
    
    def _save_analysis(self, analysis: AestheticAnalysis):
        """분석 결과 저장"""
        filename = f"{analysis.source_type}_{int(time.time())}.json"
        filepath = self.data_dir / filename
        
        data = {
            "source": analysis.source,
            "source_type": analysis.source_type,
            "title": analysis.title,
            "principles": analysis.principles_detected,
            "why_beautiful": analysis.why_beautiful,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_learning_summary(self) -> str:
        """학습 요약"""
        summary = f"📚 학습 현황\n"
        summary += f"총 분석: {len(self.learned_analyses)}개\n\n"
        
        # 원리별 패턴 수
        summary += "원리별 패턴:\n"
        for principle, patterns in self.pattern_database.items():
            summary += f"  • {principle}: {len(patterns)}개\n"
        
        return summary
    
    def suggest_creation_principles(self, concept: str, medium: Medium) -> Dict[str, float]:
        """
        창작을 위한 원리 제안
        
        학습된 패턴을 기반으로 최적의 원리 조합을 제안합니다.
        """
        return self.wisdom.suggest_for_creation(concept, medium)


# 싱글톤
_learner_instance: Optional[AestheticLearner] = None

def get_aesthetic_learner() -> AestheticLearner:
    """AestheticLearner 싱글톤"""
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = AestheticLearner()
    return _learner_instance


# 테스트
if __name__ == "__main__":
    learner = get_aesthetic_learner()
    
    # 텍스트 분석 테스트
    sample_text = """
    꽃잎이 바람에 흩날린다.
    하지만 뿌리는 대지에 단단히 박혀 있다.
    
    슬픔과 기쁨이 교차하는 순간,
    우리는 비로소 삶의 의미를 깨닫는다.
    
    그래서 다시 일어선다.
    천천히, 그러나 확실하게.
    """
    
    analysis = learner.analyze_text(sample_text, "시의 조각")
    print(analysis.why_beautiful)
    print("\n" + learner.get_learning_summary())
