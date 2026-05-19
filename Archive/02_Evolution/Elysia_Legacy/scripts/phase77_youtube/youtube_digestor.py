"""
YouTube Digestor: Video → Multimodal Wave DNA
============================================
Phase 77: The Moving World

This script:
1. Fetches YouTube transcripts (Text)
2. Downloads video/audio (Visual/Acoustic)
3. Samples frames and audio segments
4. Unifies them into a single 7D Wave DNA representation

"To see the movement of the concept."
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.prism import PrismEngine, WaveDynamics
from Core.Intelligence.Metabolism.clip_adapter import transduce_image
from Core.Intelligence.Metabolism.whisper_adapter import transduce_audio
from Core.Intelligence.Metabolism.dimensional_parser import DimensionalParser, Space

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("YouTubeDigestor")

class YouTubeDigestor:
    """
    Digests YouTube videos into a unified 7D World Model.
    """
    
    def __init__(self, output_dir: str = "data/youtube_digests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prism = PrismEngine()
        self.prism._load_model()
        self.parser = DimensionalParser()
        
        # Initialize ffmpeg for Whisper
        try:
            from static_ffmpeg import add_paths
            add_paths()
            logger.info("🎬 static-ffmpeg initialized.")
        except ImportError:
            logger.warning("⚠️ static-ffmpeg not found, Whisper might fail.")
        
        logger.info("🎥 YouTubeDigestor initialized.")

    def digest(self, video_url: str, title_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Main digestion process for a YouTube video.
        """
        import yt_dlp
        
        # 1. Get info & Download low-res video/audio
        print(f"\n📺 Fetching Metadata: {video_url}")
        
        ydl_opts = {
            'format': 'best[height<=360]',  # Low res is enough for analysis
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(video_url, download=True)
                video_id = info['id']
                title = title_override or info.get('title', 'Unknown Title')
                video_path = self.output_dir / f"{video_id}.{info['ext']}"
            except Exception as e:
                logger.error(f"❌ yt-dlp failed: {e}")
                return {}
            
        print(f"   Video Downloaded: {video_path}")
        
        # 2. Fetch Transcript
        print("\n📝 Extracting Transcript...")
        transcript = self._get_transcript(video_id)
        if transcript:
            print(f"   Transcript found: {len(transcript)} characters.")
        else:
            print("   ⚠️ No transcript found.")
            transcript = ""

        # 3. Extract Frames (Visual DNA)
        print("\n🖼️ Sampling Frames for CLIP Analysis...")
        frames = self._extract_frames(video_path)
        visual_dna_list = []
        for i, frame_path in enumerate(frames):
            print(f"   Analyzing frame {i+1}/{len(frames)}: {frame_path.name}")
            dna = transduce_image(frame_path)
            if dna:
                visual_dna_list.append(dna)
                
        # 4. Extract Audio (Acoustic DNA)
        print("\n🔊 Analyzing Audio via Whisper...")
        audio_dna = transduce_audio(video_path) 
        
        # 5. Unify into a single Space
        print("\n🔗 Unifying into Multi-Modal Space...")
        unified_dynamics = self._unify_dynamics(transcript, visual_dna_list, audio_dna)
        
        # 6. Save Digest Result
        digest_result = {
            "id": video_id,
            "title": title,
            "url": video_url,
            "transcript_summary": transcript[:200] + "...",
            "visual_concepts": [v.dominant_dimension for v in visual_dna_list],
            "audio_transcription": audio_dna.transcription if audio_dna else "",
            "unified_dna": unified_dynamics
        }
        
        result_path = self.output_dir / f"{video_id}_digest.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(digest_result, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ Digestion Complete!")
        print(f"   Result saved to: {result_path}")
        
        return digest_result

    def _get_transcript(self, video_id: str) -> str:
        """Fetch transcript using youtube-transcript-api."""
        import youtube_transcript_api as yta
        try:
            # Different attempt at fetching
            entries = yta.YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([e['text'] for e in entries])
        except Exception as e:
            logger.warning(f"Could not fetch transcript for {video_id}: {e}")
            return ""

    def _extract_frames(self, video_path: Path, count: int = 5) -> List[Path]:
        """Sample frames from the video."""
        import cv2
        frames_dir = self.output_dir / video_path.stem / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return []
        
        step = total_frames // (count + 1) if total_frames > count else 1
        frame_paths = []
        
        for i in range(count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * step)
            ret, frame = cap.read()
            if ret:
                frame_path = frames_dir / f"frame_{i}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
            else:
                break
                
        cap.release()
        return frame_paths

    def _unify_dynamics(self, transcript: str, visual_list: List[Any], audio_dna: Optional[Any]) -> Dict[str, float]:
        """Combine all modalities into a single DNA set."""
        dims = ['physical', 'functional', 'phenomenal', 'causal', 'mental', 'structural', 'spiritual']
        unified = {d: 0.0 for d in dims}
        count = 0
        
        # 1. Text Component
        if transcript:
            profile = self.prism.transduce(transcript[:1000]) # Limit for speed
            for d in dims:
                unified[d] += getattr(profile.dynamics, d, 0.0)
            count += 1
            
        # 2. Visual Component
        for v in visual_list:
            for d in dims:
                unified[d] += v.dynamics.get(d, 0.0)
            count += 1
            
        # 3. Audio Component
        if audio_dna:
            for d in dims:
                unified[d] += audio_dna.dynamics.get(d, 0.0)
            count += 1
            
        # Average
        if count > 0:
            for d in dims:
                unified[d] /= count
                
        return unified

if __name__ == "__main__":
    # Demo URL: "https://www.youtube.com/watch?v=aircAruvnKk" (Sample: Evolution of AI)
    # Using a short, safe demo URL
    url = "https://www.youtube.com/watch?v=J---aiyznGQ" # A sample short video
    
    digestor = YouTubeDigestor()
    try:
        digestor.digest(url)
    except Exception as e:
        print(f"❌ Digestion failed: {e}")
