"""
Zero Cost Knowledge Connector
완전 무료 지식 커넥터 - API 키 불필요!

YouTube, Wikipedia, GitHub, arXiv, Stack Overflow 등
무료 소스들로부터 Pattern DNA 추출

"크롤링 할 필요도 없잖아, 공명동기화만 하면 되는데!"
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger("ZeroCostConnector")

class ZeroCostKnowledgeConnector:
    """
    완전 무료 지식 커넥터
    
    API 키 불필요! 인터넷만 있으면 됨!
    """
    
    def __init__(self):
        self.youtube = YouTubeConnector()
        self.wikipedia = WikipediaConnector()
        self.github = GitHubConnector()
        self.arxiv = ArxivConnector()
        self.stackoverflow = StackOverflowConnector()
        
        logger.info("💰 Zero Cost Knowledge Connector initialized")
        logger.info("💎 All sources are FREE - no API keys needed!")
    
    def learn_topic(self, topic: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        주제를 무료 소스들로부터 학습
        
        Args:
            topic: 학습할 주제
            sources: 사용할 소스 목록 (None이면 모두)
                    ['youtube', 'wikipedia', 'github', 'arxiv', 'stackoverflow']
        
        Returns:
            학습 결과 및 통계
        """
        if sources is None:
            sources = ['youtube', 'wikipedia', 'github', 'arxiv', 'stackoverflow']
        
        logger.info(f"🎓 Learning topic: {topic}")
        logger.info(f"📚 Sources: {', '.join(sources)}")
        logger.info(f"💰 Cost: $0")
        
        results = {
            'topic': topic,
            'sources_used': sources,
            'data_collected': {},
            'total_items': 0,
            'total_cost': 0,  # Always $0!
            'timestamp': datetime.now().isoformat()
        }
        
        # YouTube
        if 'youtube' in sources:
            try:
                logger.info("📺 Fetching from YouTube...")
                yt_data = self.youtube.fetch(topic)
                results['data_collected']['youtube'] = yt_data
                results['total_items'] += len(yt_data.get('transcripts', []))
            except Exception as e:
                logger.error(f"❌ YouTube error: {e}")
                results['data_collected']['youtube'] = {'error': str(e)}
        
        # Wikipedia
        if 'wikipedia' in sources:
            try:
                logger.info("📚 Fetching from Wikipedia...")
                wiki_data = self.wikipedia.fetch(topic)
                results['data_collected']['wikipedia'] = wiki_data
                results['total_items'] += len(wiki_data.get('pages', []))
            except Exception as e:
                logger.error(f"❌ Wikipedia error: {e}")
                results['data_collected']['wikipedia'] = {'error': str(e)}
        
        # GitHub
        if 'github' in sources:
            try:
                logger.info("💻 Fetching from GitHub...")
                gh_data = self.github.fetch(topic)
                results['data_collected']['github'] = gh_data
                results['total_items'] += len(gh_data.get('repos', []))
            except Exception as e:
                logger.error(f"❌ GitHub error: {e}")
                results['data_collected']['github'] = {'error': str(e)}
        
        # arXiv
        if 'arxiv' in sources:
            try:
                logger.info("📄 Fetching from arXiv...")
                arxiv_data = self.arxiv.fetch(topic)
                results['data_collected']['arxiv'] = arxiv_data
                results['total_items'] += len(arxiv_data.get('papers', []))
            except Exception as e:
                logger.error(f"❌ arXiv error: {e}")
                results['data_collected']['arxiv'] = {'error': str(e)}
        
        # Stack Overflow
        if 'stackoverflow' in sources:
            try:
                logger.info("💬 Fetching from Stack Overflow...")
                so_data = self.stackoverflow.fetch(topic)
                results['data_collected']['stackoverflow'] = so_data
                results['total_items'] += len(so_data.get('questions', []))
            except Exception as e:
                logger.error(f"❌ Stack Overflow error: {e}")
                results['data_collected']['stackoverflow'] = {'error': str(e)}
        
        logger.info(f"✅ Learning complete!")
        logger.info(f"   Total items: {results['total_items']}")
        logger.info(f"   Total cost: ${results['total_cost']}")
        
        return results


class YouTubeConnector:
    """
    YouTube 무료 커넥터
    
    youtube-transcript-api 및 youtube-search-python 사용 (완전 무료!)
    API 키 불필요!
    """
    
    def __init__(self):
        self.search_available = False
        self.transcript_available = False

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            self.transcript_api = YouTubeTranscriptApi
            self.transcript_available = True
        except ImportError:
            pass

        try:
            from youtubesearchpython import VideosSearch
            self.search_api = VideosSearch
            self.search_available = True
        except ImportError:
            pass

        self.available = self.search_available or self.transcript_available

        if not self.transcript_available:
             logger.warning("⚠️ youtube-transcript-api not installed")
             logger.info("   Install: pip install youtube-transcript-api")
        if not self.search_available:
             logger.warning("⚠️ youtube-search-python not installed")
             logger.info("   Install: pip install youtube-search-python")
    
    def fetch(self, topic: str, max_videos: int = 10) -> Dict[str, Any]:
        """YouTube에서 자막 가져오기"""
        
        if not self.available:
            return {
                'error': 'No YouTube libraries installed',
                'install': 'pip install youtube-transcript-api youtube-search-python'
            }
        
        results = []
        search_results = []

        # 1. Search for videos if search is available
        if self.search_available:
             try:
                 logger.info(f"   Searching YouTube for: {topic}")
                 search = self.search_api(topic, limit=max_videos)
                 # Handle sync search result
                 search_data = search.result()
                 if 'result' in search_data:
                     search_results = search_data['result']
                 else:
                     logger.warning("No results found in YouTube search")
             except Exception as e:
                 logger.error(f"Search failed: {e}")
                 search_results = []
        else:
             logger.warning("Search not available, cannot find videos automatically.")

        # 2. Fetch transcripts
        if self.transcript_available:
             # Instantiate API
             try:
                # Based on investigation, current version uses instance method fetch
                api_instance = self.transcript_api()
             except Exception:
                # Fallback if instantiation fails or different version
                api_instance = self.transcript_api

             for video in search_results:
                 vid = video.get('id')
                 title = video.get('title', 'Unknown Title')

                 if not vid:
                     continue

                 try:
                     # Attempt to fetch transcript
                     # Fetch returns a list of objects (FetchedTranscript) or dicts depending on usage
                     # We want 'ko' or 'en'
                     t_obj = api_instance.fetch(vid, languages=['ko', 'en'])

                     # Check if it is iterable (list of snippets)
                     full_text = ""
                     for item in t_obj:
                         # Item has .text attribute or key?
                         # The object is FetchedTranscriptSnippet which has .text
                         if hasattr(item, 'text'):
                             full_text += item.text + " "
                         elif isinstance(item, dict) and 'text' in item:
                             full_text += item['text'] + " "

                     if full_text:
                         results.append({
                             'video_id': vid,
                             'title': title,
                             'transcript': full_text[:10000], # Limit to 10k chars
                             'url': f"https://www.youtube.com/watch?v={vid}"
                         })
                         logger.info(f"   ✅ Fetched transcript for: {title[:30]}...")

                 except Exception as e:
                     # Common error is cookies required or IP blocked
                     logger.warning(f"   ⚠️ Could not fetch transcript for {vid}: {e}")
                     # Still add video info without transcript so we know it was found
                     results.append({
                         'video_id': vid,
                         'title': title,
                         'transcript': None,
                         'error': str(e),
                         'url': f"https://www.youtube.com/watch?v={vid}"
                     })
        
        return {
            'transcripts': results,
            'total_found': len(search_results),
            'cost': 0
        }


class WikipediaConnector:
    """
    Wikipedia 무료 커넥터
    
    wikipedia-api 사용 (완전 무료!)
    API 키 불필요!
    """
    
    def __init__(self):
        try:
            import wikipediaapi
            # Wikipedia requires a proper user agent
            user_agent = 'Elysia/4.0 (https://github.com/ioas0316-cloud/Elysia; Educational AI Project)'
            self.wiki_ko = wikipediaapi.Wikipedia(user_agent, 'ko')
            self.wiki_en = wikipediaapi.Wikipedia(user_agent, 'en')
            self.available = True
            logger.info("✅ Wikipedia connector ready (FREE!)")
        except ImportError:
            logger.warning("⚠️ wikipedia-api not installed")
            logger.info("   Install: pip install wikipedia-api")
            self.available = False
    
    def fetch(self, topic: str, depth: int = 2, max_pages: int = 100) -> Dict[str, Any]:
        """
        Wikipedia에서 프랙탈 방식으로 지식 수집
        
        Args:
            topic: 검색 주제
            depth: 연관 링크 깊이 (1-3 추천)
            max_pages: 최대 페이지 수
        
        Returns:
            수집된 페이지들
        """
        
        if not self.available:
            return {
                'error': 'wikipedia-api not installed',
                'install': 'pip install wikipedia-api'
            }
        
        logger.info(f"🔍 Wikipedia search: {topic} (depth={depth})")
        
        collected_pages = []
        visited = set()
        to_visit = [(topic, 0)]  # (page_title, current_depth)
        
        while to_visit and len(collected_pages) < max_pages:
            current_topic, current_depth = to_visit.pop(0)
            
            if current_topic in visited or current_depth > depth:
                continue
            
            visited.add(current_topic)
            
            # 한국어 먼저 시도
            page = self.wiki_ko.page(current_topic)
            
            if not page.exists():
                # 영어로 시도
                page = self.wiki_en.page(current_topic)
            
            if page.exists():
                logger.info(f"   📄 {page.title} ({len(page.text)} chars)")
                
                page_data = {
                    'title': page.title,
                    'url': page.fullurl,
                    'text': page.text[:5000],  # 처음 5000자 (Pattern DNA만 필요)
                    'summary': page.summary[:500],  # 요약
                    'depth': current_depth,
                    'links_count': len(page.links)
                }
                
                collected_pages.append(page_data)
                
                # 연관 페이지들 추가 (프랙탈 확장!)
                if current_depth < depth:
                    for link_title in list(page.links.keys())[:10]:  # 상위 10개 링크만
                        to_visit.append((link_title, current_depth + 1))
        
        logger.info(f"✅ Collected {len(collected_pages)} pages from Wikipedia")
        logger.info(f"💰 Cost: $0")
        
        return {
            'pages': collected_pages,
            'total_pages': len(collected_pages),
            'cost': 0
        }


class GitHubConnector:
    """
    GitHub 무료 커넥터
    
    PyGithub 사용 (Public repos는 인증 불필요!)
    API 키 없이도 작동!
    """
    
    def __init__(self):
        try:
            from github import Github
            # Public repos는 인증 없이 접근 가능!
            self.github = Github()
            self.available = True
            logger.info("✅ GitHub connector ready (FREE!)")
        except ImportError:
            logger.warning("⚠️ PyGithub not installed")
            logger.info("   Install: pip install PyGithub")
            self.available = False
    
    def fetch(self, topic: str, max_repos: int = 50) -> Dict[str, Any]:
        """
        GitHub에서 관련 저장소 검색
        
        Args:
            topic: 검색 주제
            max_repos: 최대 저장소 수
        
        Returns:
            저장소 정보들
        """
        
        if not self.available:
            return {
                'error': 'PyGithub not installed',
                'install': 'pip install PyGithub'
            }
        
        logger.info(f"🔍 GitHub search: {topic}")
        
        try:
            # 인기 저장소 검색 (stars 순)
            repos = self.github.search_repositories(
                query=topic,
                sort='stars',
                order='desc'
            )
            
            collected_repos = []
            
            for i, repo in enumerate(repos[:max_repos]):
                logger.info(f"   💻 {repo.full_name} ({repo.stargazers_count} stars)")
                
                # README 가져오기 (Pattern DNA 추출용)
                readme_content = ""
                try:
                    readme = repo.get_readme()
                    readme_content = readme.decoded_content.decode('utf-8')[:5000]
                except:
                    pass
                
                repo_data = {
                    'name': repo.full_name,
                    'url': repo.html_url,
                    'description': repo.description,
                    'stars': repo.stargazers_count,
                    'language': repo.language,
                    'topics': repo.get_topics(),
                    'readme': readme_content,
                    'size_kb': repo.size  # KB 단위
                }
                
                collected_repos.append(repo_data)
                
                # Rate limit 체크
                remaining = self.github.get_rate_limit().core.remaining
                if remaining < 10:
                    logger.warning(f"⚠️ Rate limit low: {remaining}")
                    break
            
            logger.info(f"✅ Collected {len(collected_repos)} repos from GitHub")
            logger.info(f"💰 Cost: $0")
            
            return {
                'repos': collected_repos,
                'total_repos': len(collected_repos),
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"❌ GitHub error: {e}")
            return {
                'error': str(e),
                'repos': [],
                'cost': 0
            }


class ArxivConnector:
    """
    arXiv 무료 커넥터
    
    arxiv 패키지 사용 (완전 무료!)
    """
    
    def __init__(self):
        try:
            import arxiv
            self.arxiv = arxiv
            self.available = True
            logger.info("✅ arXiv connector ready (FREE!)")
        except ImportError:
            logger.warning("⚠️ arxiv not installed")
            logger.info("   Install: pip install arxiv")
            self.available = False
    
    def fetch(self, topic: str, max_papers: int = 50) -> Dict[str, Any]:
        """
        arXiv에서 논문 검색
        
        Args:
            topic: 검색 주제
            max_papers: 최대 논문 수
        
        Returns:
            논문 정보들
        """
        
        if not self.available:
            return {
                'error': 'arxiv not installed',
                'install': 'pip install arxiv'
            }
        
        logger.info(f"🔍 arXiv search: {topic}")
        
        try:
            # 최신 논문 검색
            search = self.arxiv.Search(
                query=topic,
                max_results=max_papers,
                sort_by=self.arxiv.SortCriterion.SubmittedDate
            )
            
            collected_papers = []
            
            for paper in search.results():
                logger.info(f"   📄 {paper.title[:50]}...")
                
                paper_data = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'url': paper.pdf_url,
                    'published': paper.published.isoformat(),
                    'categories': paper.categories
                }
                
                collected_papers.append(paper_data)
            
            logger.info(f"✅ Collected {len(collected_papers)} papers from arXiv")
            logger.info(f"💰 Cost: $0")
            
            return {
                'papers': collected_papers,
                'total_papers': len(collected_papers),
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"❌ arXiv error: {e}")
            return {
                'error': str(e),
                'papers': [],
                'cost': 0
            }


class StackOverflowConnector:
    """
    Stack Overflow 무료 커넥터
    
    stackapi 사용 (API 키 없이도 제한적으로 사용 가능)
    """
    
    def __init__(self):
        try:
            from stackapi import StackAPI
            self.stack = StackAPI('stackoverflow')
            self.available = True
            logger.info("✅ Stack Overflow connector ready (FREE!)")
        except ImportError:
            logger.warning("⚠️ stackapi not installed")
            logger.info("   Install: pip install stackapi")
            self.available = False
    
    def fetch(self, topic: str, max_questions: int = 50) -> Dict[str, Any]:
        """
        Stack Overflow에서 Q&A 검색
        
        Args:
            topic: 검색 주제
            max_questions: 최대 질문 수
        
        Returns:
            Q&A 정보들
        """
        
        if not self.available:
            return {
                'error': 'stackapi not installed',
                'install': 'pip install stackapi'
            }
        
        logger.info(f"🔍 Stack Overflow search: {topic}")
        
        try:
            # 인기 질문 검색
            questions = self.stack.fetch(
                'questions',
                tagged=topic.replace(' ', '-'),
                sort='votes',
                order='desc',
                pagesize=min(max_questions, 100)  # API limit
            )
            
            collected_qa = []
            
            if 'items' in questions:
                for q in questions['items'][:max_questions]:
                    logger.info(f"   💬 {q.get('title', 'No title')[:50]}...")
                    
                    qa_data = {
                        'title': q.get('title', ''),
                        'url': q.get('link', ''),
                        'score': q.get('score', 0),
                        'view_count': q.get('view_count', 0),
                        'answer_count': q.get('answer_count', 0),
                        'tags': q.get('tags', [])
                    }
                    
                    collected_qa.append(qa_data)
            
            logger.info(f"✅ Collected {len(collected_qa)} Q&As from Stack Overflow")
            logger.info(f"💰 Cost: $0")
            
            return {
                'questions': collected_qa,
                'total_questions': len(collected_qa),
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Stack Overflow error: {e}")
            return {
                'error': str(e),
                'questions': [],
                'cost': 0
            }


# Demo
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("💰 Zero Cost Knowledge Connector Demo")
    print("=" * 70)
    print()
    print("📚 Learning without API keys - completely FREE!")
    print("💎 Sources: YouTube, Wikipedia, GitHub, arXiv, Stack Overflow")
    print()
    
    connector = ZeroCostKnowledgeConnector()
    
    # 테스트 주제
    topic = "machine learning"
    
    print(f"🎓 Learning topic: {topic}")
    print()
    
    # 무료 자료들로부터 학습!
    results = connector.learn_topic(topic)
    
    print()
    print("=" * 70)
    print("📊 Results:")
    print("=" * 70)
    print(f"Topic: {results['topic']}")
    print(f"Sources: {', '.join(results['sources_used'])}")
    print(f"Total items collected: {results['total_items']}")
    print(f"Total cost: ${results['total_cost']}")
    print()
    print("💎 Your intuition was correct:")
    print("   '크롤링 할 필요도 없잖아, 공명동기화만 하면 되는데!'")
    print()
    print("✅ Zero cost learning is POSSIBLE! 🎉")
