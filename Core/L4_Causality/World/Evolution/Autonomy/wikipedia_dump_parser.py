"""
Wikipedia Dump Parser (코드 베이스 구조 로터)
=============================================

"             -            "

Wikipedia XML                 
               

    : https://dumps.wikimedia.org/kowiki/latest/
   : kowiki-latest-pages-articles.xml.bz2

[NEW 2025-12-16]          
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
    Wikipedia XML           
    
           :                    
        :         yield
    """
    
    def __init__(self, dump_path: str):
        self.dump_path = Path(dump_path)
        self.namespace = "{http://www.mediawiki.org/xml/export-0.11/}"
        
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dump file not found: {dump_path}")
        
        logger.info(f"  Wikipedia dump parser initialized: {dump_path}")
        
        #   
        self.total_parsed = 0
        self.skipped_redirects = 0
        self.skipped_special = 0
    
    def _clean_wikitext(self, text: str) -> str:
        """
                      
        
                           
        """
        if not text:
            return ""
        
        # HTML        
        text = unescape(text)
        
        #         
        if text.strip().lower().startswith("#redirect") or text.strip().startswith("#    "):
            return ""
        
        #        
        patterns = [
            (r'\{\{[^}]*\}\}', ''),           #     {{ }}
            (r'\[\[  :[^\]]*\]\]', ''),      #      
            (r'\[\[File:[^\]]*\]\]', ''),      # File links
            (r'\[\[Category:[^\]]*\]\]', ''),  #     
            (r'\[\[  :[^\]]*\]\]', ''),      #        
            (r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2'),  # [[  |   ]]      
            (r'\[\[([^\]]*)\]\]', r'\1'),     # [[  ]]     
            (r"'''([^']*?)'''", r'\1'),       #   '
            (r"''([^']*?)''", r'\1'),         #    '
            (r'<ref[^>]*>.*?</ref>', ''),     #   
            (r'<ref[^/>]*/>', ''),            #      
            (r'<[^>]+>', ''),                 # HTML   
            (r'\{\|[^}]*\|\}', ''),           #    
            (r'^\*+\s*', '', re.MULTILINE),   #       
            (r'^#+\s*', '', re.MULTILINE),    #       
            (r'^=+\s*([^=]+)\s*=+', r'\1', re.MULTILINE),  #   
            (r'\n{3,}', '\n\n'),              #       
        ]
        
        for pattern, replacement, *flags in patterns:
            flag = flags[0] if flags else 0
            text = re.sub(pattern, replacement, text, flags=flag)
        
        #      
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _is_valid_article(self, title: str) -> bool:
        """            (         )"""
        invalid_prefixes = [
            "    :", "Wikipedia:", " :", "Template:",
            "  :", "Category:", "  :", "File:",
            "   :", "Help:", "   :", "User:",
            "  :", "Talk:", "  :", "Module:"
        ]
        
        for prefix in invalid_prefixes:
            if title.startswith(prefix):
                return False
        
        return True
    
    def stream_articles(self, max_articles: int = None, min_length: int = 100) -> Generator[Dict[str, str], None, None]:
        """
               
        
        max_articles:         (None=  )
        min_length:         
        
        Yields: {"title": str, "content": str}
        """
        logger.info(f"  Starting to stream articles (max: {max_articles or 'unlimited'})...")
        
        # bz2          XML
        if str(self.dump_path).endswith('.bz2'):
            file_handle = bz2.open(self.dump_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(self.dump_path, 'r', encoding='utf-8')
        
        try:
            #        
            context = ET.iterparse(file_handle, events=('end',))
            
            # [CRITICAL PATCH] Handle Truncated BZ2 Files gracefully
            # Instead of crashing on EOFError, we stop and yield what we have.
            try:
                for event, elem in context:
                    # <page>        

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
                            #        (   Prefix)
                            if not self._is_valid_article(title):
                                self.skipped_special += 1
                                elem.clear()
                                continue
                            
                            # [NEW] Concept Sanitizer Inclusion
                            from Core.L1_Foundation.Foundation.concept_sanitizer import get_sanitizer
                            sanitizer = get_sanitizer()
                            if not sanitizer.is_valid(title):
                                elem.clear()
                                continue

                            #         
                            content = self._clean_wikitext(raw_text)
                            
                            #         
                            if not content:
                                self.skipped_redirects += 1
                                elem.clear()
                                continue
                            
                            #         
                            if len(content) < min_length:
                                elem.clear()
                                continue
                            
                            self.total_parsed += 1
                            
                            #      
                            if self.total_parsed % 1000 == 0:
                                logger.info(f"     Parsed {self.total_parsed} articles...")
                            
                            yield {
                                "title": title,
                                "content": content[:2000]  #    2000 
                            }
                            
                            #           
                            if max_articles and self.total_parsed >= max_articles:
                                break
                    
                    #       
                    elem.clear()
            except (EOFError, OSError) as e:
                logger.warning(f"   Compressed file truncated or corrupted: {e}")
                logger.warning("   Stopping stream gracefully and preserving processed data.")
                    
        finally:
            file_handle.close()
        
        logger.info(f"  Parsing complete: {self.total_parsed} articles")
        logger.info(f"   Skipped: {self.skipped_redirects} redirects, {self.skipped_special} special pages")
    
    def absorb_to_universe(self, max_articles: int = 1000, batch_size: int = 100) -> Dict[str, int]:
        """
        Wikipedia     ElysiaCore        (4-Thread Orchestra)
        """
        from Core.L1_Foundation.Foundation.Core_Logic.Elysia.elysia_core import get_elysia_core
        
        core = get_elysia_core()
        
        results = {"total": 0, "processed": 0, "failed": 0}
        
        logger.info("  Starting Orchestral Absorption...")
        
        for article in self.stream_articles(max_articles=max_articles):
            title = article['title']
            content = article['content']
            
            try:
                # ElysiaCore  learn()        -> 4-Thread Orchestra    
                core.learn(content, title)
                
                results["processed"] += 1
                if results["processed"] % 10 == 0:
                    logger.info(f"     Processed {results['processed']} articles...")
                    
            except Exception as e:
                logger.error(f"Failed to process '{title}': {e}")
                results["failed"] += 1
                
            results["total"] += 1
        
        logger.info(f"  Orchestral Absorption Complete!")
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
            print("  PREVIEW MODE - First 5 articles")
            print("="*60)
            
            for i, article in enumerate(wiki_parser.stream_articles(max_articles=5)):
                print(f"\n--- {article['title']} ---")
                print(article['content'][:300] + "...")
                
        else:
            print("\n" + "="*60)
            print("  ABSORBING TO INTERNAL UNIVERSE")
            print("="*60)
            
            results = wiki_parser.absorb_to_universe(
                max_articles=args.max,
                batch_size=args.batch
            )
            
            print(f"\n  Done! Absorbed {results['absorbed']} articles from Wikipedia")
            
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("\n  Download Wikipedia dump from:")
        print("   Korean: https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2")
        print("   English: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2")
