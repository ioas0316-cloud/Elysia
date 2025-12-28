"""
Wikipedia Dump Parser (위키피디아 덤프 파서)
=============================================

"진짜 지식의 대량 흡수 - 크롤링 없이 로컬에서"

Wikipedia XML 덤프 파일을 스트리밍 파싱하여
메모리 효율적으로 대량 흡수

다운로드: https://dumps.wikimedia.org/kowiki/latest/
파일명: kowiki-latest-pages-articles.xml.bz2

[NEW 2025-12-16] 진짜 데이터 파서
"""

import os
import sys
import bz2
import re
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Optional
import xml.etree.ElementTree as ET
from html import unescape

sys.path.insert(0, "c:\\Elysia")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WikipediaDumpParser")


class WikipediaDumpParser:
    """
    Wikipedia XML 덤프 스트리밍 파서
    
    메모리 효율적: 전체 파일을 메모리에 로드하지 않음
    스트리밍: 문서를 하나씩 yield
    """
    
    def __init__(self, dump_path: str):
        self.dump_path = Path(dump_path)
        self.namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
        
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dump file not found: {dump_path}")
        
        logger.info(f"📚 Wikipedia dump parser initialized: {dump_path}")
        
        # 통계
        self.total_parsed = 0
        self.skipped_redirects = 0
        self.skipped_special = 0
    
    def _clean_wikitext(self, text: str) -> str:
        """
        위키텍스트에서 마크업 제거
        
        완벽하지 않지만 대부분의 내용 추출
        """
        if not text:
            return ""
        
        # HTML 엔티티 디코딩
        text = unescape(text)
        
        # 리다이렉트 체크
        if text.strip().lower().startswith("#redirect") or text.strip().startswith("#넘겨주기"):
            return ""
        
        # 제거할 패턴들
        patterns = [
            (r'\{\{[^}]*\}\}', ''),           # 템플릿 {{ }}
            (r'\[\[파일:[^\]]*\]\]', ''),      # 파일 링크
            (r'\[\[File:[^\]]*\]\]', ''),      # File links
            (r'\[\[Category:[^\]]*\]\]', ''),  # 카테고리
            (r'\[\[분류:[^\]]*\]\]', ''),      # 한글 카테고리
            (r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2'),  # [[링크|텍스트]] → 텍스트
            (r'\[\[([^\]]*)\]\]', r'\1'),     # [[링크]] → 링크
            (r"'''([^']*?)'''", r'\1'),       # 볼드
            (r"''([^']*?)''", r'\1'),         # 이탤릭
            (r'<ref[^>]*>.*?</ref>', ''),     # 참조
            (r'<ref[^/>]*/>', ''),            # 단일 참조
            (r'<[^>]+>', ''),                 # HTML 태그
            (r'\{\|[^}]*\|\}', ''),           # 테이블
            (r'^\*+\s*', '', re.MULTILINE),   # 리스트 마커
            (r'^#+\s*', '', re.MULTILINE),    # 번호 리스트
            (r'^=+\s*([^=]+)\s*=+', r'\1', re.MULTILINE),  # 헤더
            (r'\n{3,}', '\n\n'),              # 과다 줄바꿈
        ]
        
        for pattern, replacement, *flags in patterns:
            flag = flags[0] if flags else 0
            text = re.sub(pattern, replacement, text, flags=flag)
        
        # 공백 정리
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _is_valid_article(self, title: str) -> bool:
        """유효한 문서인지 확인 (특수 페이지 제외)"""
        invalid_prefixes = [
            "위키백과:", "Wikipedia:", "틀:", "Template:",
            "분류:", "Category:", "파일:", "File:",
            "도움말:", "Help:", "사용자:", "User:",
            "토론:", "Talk:", "모듈:", "Module:"
        ]
        
        for prefix in invalid_prefixes:
            if title.startswith(prefix):
                return False
        
        return True
    
    def stream_articles(self, max_articles: int = None, min_length: int = 100) -> Generator[Dict[str, str], None, None]:
        """
        문서 스트리밍
        
        max_articles: 최대 문서 수 (None=전체)
        min_length: 최소 본문 길이
        
        Yields: {"title": str, "content": str}
        """
        logger.info(f"🔄 Starting to stream articles (max: {max_articles or 'unlimited'})...")
        
        # bz2 압축 또는 일반 XML
        if str(self.dump_path).endswith('.bz2'):
            file_handle = bz2.open(self.dump_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(self.dump_path, 'r', encoding='utf-8')
        
        try:
            # 스트리밍 파싱
            context = ET.iterparse(file_handle, events=('end',))
            
            # [CRITICAL PATCH] Handle Truncated BZ2 Files gracefully
            # Instead of crashing on EOFError, we stop and yield what we have.
            try:
                for event, elem in context:
                    # <page> 태그 완료 시

                    # [ROBUST PATCH] Namespace-agnostic tag check
                    tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    
                    if tag_name == "page":
                        title = ""
                        raw_text = ""
                        
                        # Navigate children manually to find title and text
                        # Structure: page -> title, page -> revision -> text
                        for child in elem:
                            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                            if child_tag == "title":
                                title = child.text or ""
                            elif child_tag == "revision":
                                for sub in child:
                                    sub_tag = sub.tag.split('}')[-1] if '}' in sub.tag else sub.tag
                                    if sub_tag == "text":
                                        raw_text = sub.text or ""
                        
                        if title and raw_text:
                            # 유효성 검사 (기존 Prefix)
                            if not self._is_valid_article(title):
                                self.skipped_special += 1
                                elem.clear()
                                continue
                            
                            # [NEW] Concept Sanitizer Inclusion
                            from Core._01_Foundation._05_Governance.Foundation.concept_sanitizer import get_sanitizer
                            sanitizer = get_sanitizer()
                            if not sanitizer.is_valid(title):
                                elem.clear()
                                continue

                            # 위키텍스트 정제
                            content = self._clean_wikitext(raw_text)
                            
                            # 리다이렉트 스킵
                            if not content:
                                self.skipped_redirects += 1
                                elem.clear()
                                continue
                            
                            # 최소 길이 체크
                            if len(content) < min_length:
                                elem.clear()
                                continue
                            
                            self.total_parsed += 1
                            
                            # 진행 로그
                            if self.total_parsed % 1000 == 0:
                                logger.info(f"   📄 Parsed {self.total_parsed} articles...")
                            
                            yield {
                                "title": title,
                                "content": content[:2000]  # 최대 2000자
                            }
                            
                            # 최대 문서 수 체크
                            if max_articles and self.total_parsed >= max_articles:
                                break
                    
                    # 메모리 정리
                    elem.clear()
            except (EOFError, OSError) as e:
                logger.warning(f"⚠️ Compressed file truncated or corrupted: {e}")
                logger.warning("   Stopping stream gracefully and preserving processed data.")
                    
        finally:
            file_handle.close()
        
        logger.info(f"✅ Parsing complete: {self.total_parsed} articles")
        logger.info(f"   Skipped: {self.skipped_redirects} redirects, {self.skipped_special} special pages")
    
    def absorb_to_universe(self, max_articles: int = 1000, batch_size: int = 100) -> Dict[str, int]:
        """
        Wikipedia 덤프를 ElysiaCore를 통해 흡수 (4-Thread Orchestra)
        """
        from Core._01_Foundation.01_Core_Logic.Elysia.elysia_core import get_elysia_core
        
        core = get_elysia_core()
        
        results = {"total": 0, "processed": 0, "failed": 0}
        
        logger.info("🎻 Starting Orchestral Absorption...")
        
        for article in self.stream_articles(max_articles=max_articles):
            title = article['title']
            content = article['content']
            
            try:
                # ElysiaCore의 learn() 메소드 호출 -> 4-Thread Orchestra 트리거
                core.learn(content, title)
                
                results["processed"] += 1
                if results["processed"] % 10 == 0:
                    logger.info(f"   🎵 Processed {results['processed']} articles...")
                    
            except Exception as e:
                logger.error(f"Failed to process '{title}': {e}")
                results["failed"] += 1
                
            results["total"] += 1
        
        logger.info(f"🎉 Orchestral Absorption Complete!")
        logger.info(f"   Total: {results['total']}, Processed: {results['processed']}, Failed: {results['failed']}")
        
        return results

# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wikipedia Dump Parser")
    parser.add_argument("dump_path", help="Path to Wikipedia XML dump (can be .bz2)")
    parser.add_argument("--max", type=int, default=1000, help="Max articles to process")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    parser.add_argument("--preview", action="store_true", help="Preview first 5 articles only")
    
    args = parser.parse_args()
    
    try:
        wiki_parser = WikipediaDumpParser(args.dump_path)
        
        if args.preview:
            print("\n" + "="*60)
            print("📖 PREVIEW MODE - First 5 articles")
            print("="*60)
            
            for i, article in enumerate(wiki_parser.stream_articles(max_articles=5)):
                print(f"\n--- {article['title']} ---")
                print(article['content'][:300] + "...")
                
        else:
            print("\n" + "="*60)
            print("🧠 ABSORBING TO INTERNAL UNIVERSE")
            print("="*60)
            
            results = wiki_parser.absorb_to_universe(
                max_articles=args.max,
                batch_size=args.batch
            )
            
            print(f"\n✅ Done! Absorbed {results['absorbed']} articles from Wikipedia")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Download Wikipedia dump from:")
        print("   Korean: https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2")
        print("   English: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2")
