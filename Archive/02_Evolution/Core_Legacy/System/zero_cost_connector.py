"""
Zero Cost Knowledge Connector
             - API      !

YouTube, Wikipedia, GitHub, arXiv, Stack Overflow  
          Pattern DNA   

"             ,              !"
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
                
    
    API      !           !
    """
    
    def __init__(self):
        self.youtube = YouTubeConnector()
        self.wikipedia = WikipediaConnector()
        self.github = GitHubConnector()
        self.arxiv = ArxivConnector()
        self.stackoverflow = StackOverflowConnector()
        
        logger.info("  Zero Cost Knowledge Connector initialized")
        logger.info("  All sources are FREE - no API keys needed!")
    
    def learn_topic(self, topic: str, sources: List[str] = None) -> Dict[str, Any]:
        """
                        
        
        Args:
            topic:       
            sources:           (None     )
                    ['youtube', 'wikipedia', 'github', 'arxiv', 'stackoverflow']
        
        Returns:
                      
        """
        if sources is None:
            sources = ['youtube', 'wikipedia', 'github', 'arxiv', 'stackoverflow']
        
        logger.info(f"  Learning topic: {topic}")
        logger.info(f"  Sources: {', '.join(sources)}")
        logger.info(f"  Cost: $0")
        
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
                logger.info("  Fetching from YouTube...")
                yt_data = self.youtube.fetch(topic)
                results['data_collected']['youtube'] = yt_data
                results['total_items'] += len(yt_data.get('transcripts', []))
            except Exception as e:
                logger.error(f"  YouTube error: {e}")
                results['data_collected']['youtube'] = {'error': str(e)}
        
        # Wikipedia
        if 'wikipedia' in sources:
            try:
                logger.info("  Fetching from Wikipedia...")
                wiki_data = self.wikipedia.fetch(topic)
                results['data_collected']['wikipedia'] = wiki_data
                results['total_items'] += len(wiki_data.get('pages', []))
            except Exception as e:
                logger.error(f"  Wikipedia error: {e}")
                results['data_collected']['wikipedia'] = {'error': str(e)}
        
        # GitHub
        if 'github' in sources:
            try:
                logger.info("  Fetching from GitHub...")
                gh_data = self.github.fetch(topic)
                results['data_collected']['github'] = gh_data
                results['total_items'] += len(gh_data.get('repos', []))
            except Exception as e:
                logger.error(f"  GitHub error: {e}")
                results['data_collected']['github'] = {'error': str(e)}
        
        # arXiv
        if 'arxiv' in sources:
            try:
                logger.info("  Fetching from arXiv...")
                arxiv_data = self.arxiv.fetch(topic)
                results['data_collected']['arxiv'] = arxiv_data
                results['total_items'] += len(arxiv_data.get('papers', []))
            except Exception as e:
                logger.error(f"  arXiv error: {e}")
                results['data_collected']['arxiv'] = {'error': str(e)}
        
        # Stack Overflow
        if 'stackoverflow' in sources:
            try:
                logger.info("  Fetching from Stack Overflow...")
                so_data = self.stackoverflow.fetch(topic)
                results['data_collected']['stackoverflow'] = so_data
                results['total_items'] += len(so_data.get('questions', []))
            except Exception as e:
                logger.error(f"  Stack Overflow error: {e}")
                results['data_collected']['stackoverflow'] = {'error': str(e)}
        
        logger.info(f"  Learning complete!")
        logger.info(f"   Total items: {results['total_items']}")
        logger.info(f"   Total cost: ${results['total_cost']}")
        
        return results


class YouTubeConnector:
    """
    YouTube       
    
    youtube-transcript-api & youtube-search-python    (     !)
    API      !
    """
    
    def __init__(self):
        self.available = False
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            self.api = YouTubeTranscriptApi
            self.available = True
        except ImportError:
            logger.warning("   youtube-transcript-api not installed")
            logger.info("   Install: pip install youtube-transcript-api")

        try:
            # Monkey Patch for youtube-search-python compatibility with httpx >= 0.28.0
            # The library tries to pass 'proxies' directly to httpx.post, which was removed.
            from youtubesearchpython.core.requests import RequestCore
            from youtubesearchpython.core.constants import userAgent
            import httpx

            def patched_syncPostRequest(self) -> httpx.Response:
                # Use Client for proxy support (renamed from proxies in older versions)
                # httpx 0.28.1 uses 'proxy' argument
                proxy = self.proxy.get("https://") or self.proxy.get("http://")

                with httpx.Client(proxy=proxy) as client:
                    return client.post(
                        self.url,
                        headers={"User-Agent": userAgent},
                        json=self.data,
                        timeout=self.timeout
                    )

            def patched_syncGetRequest(self) -> httpx.Response:
                proxy = self.proxy.get("https://") or self.proxy.get("http://")

                with httpx.Client(proxy=proxy) as client:
                    return client.get(
                        self.url,
                        headers={"User-Agent": userAgent},
                        timeout=self.timeout,
                        cookies={'CONSENT': 'YES+1'}
                    )

            RequestCore.syncPostRequest = patched_syncPostRequest
            RequestCore.syncGetRequest = patched_syncGetRequest

            from youtubesearchpython import VideosSearch
            self.search_api = VideosSearch
        except ImportError:
            logger.warning("   youtube-search-python not installed")
            logger.info("   Install: pip install youtube-search-python")
            self.available = False
    
    def fetch(self, topic: str, max_videos: int = 10) -> Dict[str, Any]:
        """
        YouTube                   

        Args:
            topic:       
            max_videos:         

        Returns:
                        (Pattern DNA)
        """
        
        if not self.available:
            return {
                'error': 'Required packages not installed',
                'install': 'pip install youtube-transcript-api youtube-search-python'
            }

        logger.info(f"  YouTube search: {topic}")
        
        try:
            # 1.        (  )
            videos_search = self.search_api(topic, limit=max_videos)
            results = videos_search.result()

            collected_videos = []

            for video in results.get('result', []):
                video_id = video.get('id')
                title = video.get('title')
                link = video.get('link')
                duration = video.get('duration')
                view_count = video.get('viewCount', {}).get('text')

                logger.info(f"     Found: {title[:40]}... ({duration})")

                video_data = {
                    'id': video_id,
                    'title': title,
                    'link': link,
                    'duration': duration,
                    'views': view_count,
                    'transcript': None,
                    'transcript_language': None
                }

                # 2.         (  )
                try:
                    # youtube-transcript-api 1.2.3 Compatibility
                    # API version check: list_transcripts vs list
                    if hasattr(self.api, 'list_transcripts'):
                        # Newer API
                        transcript_list = self.api.list_transcripts(video_id)
                        transcript = None
                        lang = None

                        try:
                            transcript = transcript_list.find_transcript(['ko'])
                            lang = 'ko'
                        except:
                            try:
                                transcript = transcript_list.find_transcript(['en'])
                                lang = 'en'
                            except:
                                try:
                                    transcript = transcript_list.find_generated_transcript(['ko', 'en'])
                                    lang = transcript.language_code
                                except:
                                    pass

                        if transcript:
                            full_text = " ".join([t['text'] for t in transcript.fetch()])
                            video_data['transcript'] = full_text[:5000]
                            video_data['transcript_language'] = lang
                            logger.info(f"        Transcript found ({lang}, {len(full_text)} chars)")

                    else:
                        # Older API (0.2.x or 1.2.3 ?) - 'fetch' is static or instance method?
                        # In 1.2.3: fetch(self, video_id, languages=['en'])
                        # However, typical usage is YouTubeTranscriptApi.get_transcript(video_id) in modern versions
                        # or YouTubeTranscriptApi.fetch(video_id) in older ones?

                        # Let's try calling fetch directly if it's static/class method, or via instance
                        # The error shows type object 'YouTubeTranscriptApi' has no attribute 'list_transcripts'
                        # which suggests we are using the class directly.

                        # Note: In 1.2.3, fetch and list seem to be instance methods if initialized,
                        # OR if api is the class itself, we need to instantiate it?
                        # The code `self.api = YouTubeTranscriptApi` assigns the CLASS.

                        # Try to use .get_transcript if available (modern wrapper), or .fetch
                        # But inspect showed .fetch(self, ...). So we need an instance.

                        api_instance = self.api()
                        # Try Korean first
                        try:
                            # fetch signature: (video_id, languages=('en',), preserve_formatting=False)
                            # It returns a list of dicts directly in older versions?
                            # Inspect said it returns FetchedTranscript object.

                            # Let's try simple fetch with languages list
                            transcript_data = api_instance.fetch(video_id, languages=['ko', 'en'])

                            # If it returns an object that needs .fetch(), call it.
                            # If it returns list of dicts, join them.
                            # Based on name FetchedTranscript, it might be the data itself or object.

                            # Let's assume it returns something iterable or with .fetch()
                            # If it is a list of dicts:
                            if isinstance(transcript_data, list):
                                full_text = " ".join([t['text'] for t in transcript_data])
                            else:
                                # It's an object
                                full_text = str(transcript_data)

                            video_data['transcript'] = full_text[:5000]
                            video_data['transcript_language'] = 'ko/en'
                            logger.info(f"        Transcript found (legacy API)")

                        except Exception as e:
                            logger.warning(f"         Transcript fetch failed: {e}")

                except Exception as e:
                    logger.warning(f"         Transcript error: {e}")

                collected_videos.append(video_data)

            logger.info(f"  Collected {len(collected_videos)} videos from YouTube")
            logger.info(f"  Cost: $0")

            return {
                'transcripts': collected_videos, #                ,                
                'videos': collected_videos,
                'total_videos': len(collected_videos),
                'cost': 0
            }

        except Exception as e:
            logger.error(f"  YouTube search error: {e}")
            return {
                'error': str(e),
                'transcripts': [],
                'cost': 0
            }


class WikipediaConnector:
    """
    Wikipedia       
    
    wikipedia-api    (     !)
    API      !
    """
    
    def __init__(self):
        try:
            import wikipediaapi
            # Wikipedia requires a proper user agent
            user_agent = 'Elysia/4.0 (https://github.com/ioas0316-cloud/Elysia; Educational AI Project)'
            self.wiki_ko = wikipediaapi.Wikipedia(user_agent, 'ko')
            self.wiki_en = wikipediaapi.Wikipedia(user_agent, 'en')
            self.available = True
            logger.info("  Wikipedia connector ready (FREE!)")
        except ImportError:
            logger.warning("   wikipedia-api not installed")
            logger.info("   Install: pip install wikipedia-api")
            self.available = False
    
    def fetch(self, topic: str, depth: int = 2, max_pages: int = 100) -> Dict[str, Any]:
        """
        Wikipedia                 
        
        Args:
            topic:      
            depth:          (1-3   )
            max_pages:         
        
        Returns:
                    
        """
        
        if not self.available:
            return {
                'error': 'wikipedia-api not installed',
                'install': 'pip install wikipedia-api'
            }
        
        logger.info(f"  Wikipedia search: {topic} (depth={depth})")
        
        collected_pages = []
        visited = set()
        to_visit = [(topic, 0)]  # (page_title, current_depth)
        
        while to_visit and len(collected_pages) < max_pages:
            current_topic, current_depth = to_visit.pop(0)
            
            if current_topic in visited or current_depth > depth:
                continue
            
            visited.add(current_topic)
            
            #          
            page = self.wiki_ko.page(current_topic)
            
            if not page.exists():
                #       
                page = self.wiki_en.page(current_topic)
            
            if page.exists():
                logger.info(f"     {page.title} ({len(page.text)} chars)")
                
                page_data = {
                    'title': page.title,
                    'url': page.fullurl,
                    'text': page.text[:5000],  #    5000  (Pattern DNA    )
                    'summary': page.summary[:500],  #   
                    'depth': current_depth,
                    'links_count': len(page.links)
                }
                
                collected_pages.append(page_data)
                
                #            (      !)
                if current_depth < depth:
                    for link_title in list(page.links.keys())[:10]:  #    10     
                        to_visit.append((link_title, current_depth + 1))
        
        logger.info(f"  Collected {len(collected_pages)} pages from Wikipedia")
        logger.info(f"  Cost: $0")
        
        return {
            'pages': collected_pages,
            'total_pages': len(collected_pages),
            'cost': 0
        }


class GitHubConnector:
    """
    GitHub       
    
    PyGithub    (Public repos        !)
    API         !
    """
    
    def __init__(self):
        try:
            from github import Github
            # Public repos             !
            self.github = Github()
            self.available = True
            logger.info("  GitHub connector ready (FREE!)")
        except ImportError:
            logger.warning("   PyGithub not installed")
            logger.info("   Install: pip install PyGithub")
            self.available = False
    
    def fetch(self, topic: str, max_repos: int = 50) -> Dict[str, Any]:
        """
        GitHub            
        
        Args:
            topic:      
            max_repos:         
        
        Returns:
                   
        """
        
        if not self.available:
            return {
                'error': 'PyGithub not installed',
                'install': 'pip install PyGithub'
            }
        
        logger.info(f"  GitHub search: {topic}")
        
        try:
            #           (stars  )
            repos = self.github.search_repositories(
                query=topic,
                sort='stars',
                order='desc'
            )
            
            collected_repos = []
            
            for i, repo in enumerate(repos[:max_repos]):
                logger.info(f"     {repo.full_name} ({repo.stargazers_count} stars)")
                
                # README      (Pattern DNA    )
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
                    'size_kb': repo.size  # KB   
                }
                
                collected_repos.append(repo_data)
                
                # Rate limit   
                remaining = self.github.get_rate_limit().core.remaining
                if remaining < 10:
                    logger.warning(f"   Rate limit low: {remaining}")
                    break
            
            logger.info(f"  Collected {len(collected_repos)} repos from GitHub")
            logger.info(f"  Cost: $0")
            
            return {
                'repos': collected_repos,
                'total_repos': len(collected_repos),
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"  GitHub error: {e}")
            return {
                'error': str(e),
                'repos': [],
                'cost': 0
            }


class ArxivConnector:
    """
    arXiv       
    
    arxiv        (     !)
    """
    
    def __init__(self):
        try:
            import arxiv
            self.arxiv = arxiv
            self.available = True
            logger.info("  arXiv connector ready (FREE!)")
        except ImportError:
            logger.warning("   arxiv not installed")
            logger.info("   Install: pip install arxiv")
            self.available = False
    
    def fetch(self, topic: str, max_papers: int = 50) -> Dict[str, Any]:
        """
        arXiv        
        
        Args:
            topic:      
            max_papers:        
        
        Returns:
                  
        """
        
        if not self.available:
            return {
                'error': 'arxiv not installed',
                'install': 'pip install arxiv'
            }
        
        logger.info(f"  arXiv search: {topic}")
        
        try:
            #         
            search = self.arxiv.Search(
                query=topic,
                max_results=max_papers,
                sort_by=self.arxiv.SortCriterion.SubmittedDate
            )
            
            collected_papers = []
            
            for paper in search.results():
                logger.info(f"     {paper.title[:50]}...")
                
                paper_data = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'url': paper.pdf_url,
                    'published': paper.published.isoformat(),
                    'categories': paper.categories
                }
                
                collected_papers.append(paper_data)
            
            logger.info(f"  Collected {len(collected_papers)} papers from arXiv")
            logger.info(f"  Cost: $0")
            
            return {
                'papers': collected_papers,
                'total_papers': len(collected_papers),
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"  arXiv error: {e}")
            return {
                'error': str(e),
                'papers': [],
                'cost': 0
            }


class StackOverflowConnector:
    """
    Stack Overflow       
    
    stackapi    (API                  )
    """
    
    def __init__(self):
        try:
            from stackapi import StackAPI
            self.stack = StackAPI('stackoverflow')
            self.available = True
            logger.info("  Stack Overflow connector ready (FREE!)")
        except ImportError:
            logger.warning("   stackapi not installed")
            logger.info("   Install: pip install stackapi")
            self.available = False
    
    def fetch(self, topic: str, max_questions: int = 50) -> Dict[str, Any]:
        """
        Stack Overflow   Q&A   
        
        Args:
            topic:      
            max_questions:        
        
        Returns:
            Q&A    
        """
        
        if not self.available:
            return {
                'error': 'stackapi not installed',
                'install': 'pip install stackapi'
            }
        
        logger.info(f"  Stack Overflow search: {topic}")
        
        try:
            #         
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
                    logger.info(f"     {q.get('title', 'No title')[:50]}...")
                    
                    qa_data = {
                        'title': q.get('title', ''),
                        'url': q.get('link', ''),
                        'score': q.get('score', 0),
                        'view_count': q.get('view_count', 0),
                        'answer_count': q.get('answer_count', 0),
                        'tags': q.get('tags', [])
                    }
                    
                    collected_qa.append(qa_data)
            
            logger.info(f"  Collected {len(collected_qa)} Q&As from Stack Overflow")
            logger.info(f"  Cost: $0")
            
            return {
                'questions': collected_qa,
                'total_questions': len(collected_qa),
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"  Stack Overflow error: {e}")
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
    print("  Zero Cost Knowledge Connector Demo")
    print("=" * 70)
    print()
    print("  Learning without API keys - completely FREE!")
    print("  Sources: YouTube, Wikipedia, GitHub, arXiv, Stack Overflow")
    print()
    
    connector = ZeroCostKnowledgeConnector()
    
    #       
    topic = "machine learning"
    
    print(f"  Learning topic: {topic}")
    print()
    
    #             !
    results = connector.learn_topic(topic)
    
    print()
    print("=" * 70)
    print("  Results:")
    print("=" * 70)
    print(f"Topic: {results['topic']}")
    print(f"Sources: {', '.join(results['sources_used'])}")
    print(f"Total items collected: {results['total_items']}")
    print(f"Total cost: ${results['total_cost']}")
    print()
    print("  Your intuition was correct:")
    print("   '             ,              !'")
    print()
    print("  Zero cost learning is POSSIBLE!  ")
