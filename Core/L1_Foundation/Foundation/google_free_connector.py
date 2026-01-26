"""
Google Free Services Connector
                  

"                                  "
             !  

      :
- YouTube Data API (10,000 units/day   )
- Google Books API (1,000 requests/day   )
- Google Gemini API (1,500 requests/day   !)  
-        Google API         !

  : $0
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger("GoogleFreeConnector")

class GoogleFreeServicesConnector:
    """
                    
    
        Google              !
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
           
        
        Args:
            api_key: Google API   (              )
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('YOUTUBE_API_KEY')
        
        #          
        self.youtube = GoogleYouTubeConnector(self.api_key)
        self.books = GoogleBooksConnector(self.api_key)
        self.gemini = GoogleGeminiConnector(api_key or os.getenv('GEMINI_API_KEY'))
        
        logger.info("  Google Free Services Connector initialized")
        logger.info("  All services FREE with Google account!")
        
        #          
        self.quota = {
            'youtube': {'used': 0, 'limit': 10000, 'unit': 'units'},
            'books': {'used': 0, 'limit': 1000, 'unit': 'requests'},
            'gemini': {'used': 0, 'limit': 1500, 'unit': 'requests'}
        }
    
    def learn_topic_with_google(self, topic: str, use_services: List[str] = None) -> Dict[str, Any]:
        """
                      
        
        Args:
            topic:      
            use_services:            ['youtube', 'books', 'gemini']
        
        Returns:
                 
        """
        if use_services is None:
            use_services = ['youtube', 'books']  # Gemini     
        
        logger.info(f"  Learning: {topic}")
        logger.info(f"  Using Google services: {', '.join(use_services)}")
        logger.info(f"  Cost: $0")
        
        results = {
            'topic': topic,
            'services_used': use_services,
            'data': {},
            'quota_used': {},
            'total_cost': 0,  # Always $0!
            'timestamp': datetime.now().isoformat()
        }
        
        # YouTube
        if 'youtube' in use_services and self.youtube.available:
            try:
                logger.info("  Searching YouTube...")
                yt_data = self.youtube.search_videos(topic, max_results=10)
                results['data']['youtube'] = yt_data
                results['quota_used']['youtube'] = yt_data.get('quota_used', 0)
            except Exception as e:
                logger.error(f"  YouTube error: {e}")
                results['data']['youtube'] = {'error': str(e)}
        
        # Google Books
        if 'books' in use_services and self.books.available:
            try:
                logger.info("  Searching Google Books...")
                books_data = self.books.search_books(topic, max_results=10)
                results['data']['books'] = books_data
                results['quota_used']['books'] = books_data.get('quota_used', 0)
            except Exception as e:
                logger.error(f"  Books error: {e}")
                results['data']['books'] = {'error': str(e)}
        
        # Gemini (    - LLM   )
        if 'gemini' in use_services and self.gemini.available:
            try:
                logger.info("  Enhancing with Gemini...")
                #          Gemini    /  
                context = self._prepare_context(results['data'])
                gemini_response = self.gemini.generate_content(
                    f"Summarize and explain the key concepts about: {topic}\n\nContext: {context}"
                )
                results['data']['gemini'] = gemini_response
                results['quota_used']['gemini'] = 1
            except Exception as e:
                logger.error(f"  Gemini error: {e}")
                results['data']['gemini'] = {'error': str(e)}
        
        logger.info("  Learning complete!")
        logger.info(f"  Total cost: $0")
        
        return results
    
    def _prepare_context(self, data: Dict[str, Any]) -> str:
        """                 """
        context_parts = []
        
        if 'youtube' in data and 'videos' in data['youtube']:
            videos = data['youtube']['videos'][:3]  #    3 
            for video in videos:
                context_parts.append(f"Video: {video.get('title', '')}")
        
        if 'books' in data and 'books' in data['books']:
            books = data['books']['books'][:3]  #    3 
            for book in books:
                title = book.get('title', '')
                desc = book.get('description', '')[:200]
                context_parts.append(f"Book: {title} - {desc}")
        
        return "\n\n".join(context_parts)


class GoogleYouTubeConnector:
    """YouTube Data API     (  !)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.available = False
        
        if not self.api_key:
            logger.warning("   YouTube API key not provided")
            logger.info("   Get free key: https://console.cloud.google.com")
            return
        
        try:
            from googleapiclient.discovery import build
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            self.available = True
            logger.info("  YouTube Data API ready (FREE 10,000 units/day!)")
        except ImportError:
            logger.warning("   google-api-python-client not installed")
            logger.info("   Install: pip install google-api-python-client")
        except Exception as e:
            logger.error(f"  YouTube API error: {e}")
    
    def search_videos(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        YouTube       
        
        Args:
            query:    
            max_results:         (   10 )
        
        Returns:
                            
        """
        if not self.available:
            return {
                'error': 'YouTube API not available',
                'setup': 'Get free API key at https://console.cloud.google.com'
            }
        
        try:
            #        (100 units per search)
            request = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=max_results,
                order='relevance'
            )
            
            response = request.execute()
            
            videos = []
            for item in response.get('items', []):
                video_data = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                }
                videos.append(video_data)
                
                logger.info(f"     {video_data['title'][:60]}...")
            
            logger.info(f"  Found {len(videos)} videos")
            logger.info(f"  Quota used: ~100 units (10,000 daily limit)")
            logger.info(f"  Cost: $0")
            
            return {
                'videos': videos,
                'total_results': response.get('pageInfo', {}).get('totalResults', 0),
                'quota_used': 100,  #     100 units
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"  YouTube search error: {e}")
            return {'error': str(e), 'videos': []}


class GoogleBooksConnector:
    """Google Books API     (  !)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.api_url = "https://www.googleapis.com/books/v1/volumes"
        self.available = True  # API                  
        
        if api_key:
            logger.info("  Google Books API ready (FREE 1,000 requests/day!)")
        else:
            logger.info("   Google Books API (limited without key)")
            logger.info("   Get free key for more quota: https://console.cloud.google.com")
    
    def search_books(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Google Books   
        
        Args:
            query:    
            max_results:        
        
        Returns:
                
        """
        try:
            import requests
            
            params = {
                'q': query,
                'maxResults': max_results,
                'printType': 'books'
            }
            
            if self.api_key:
                params['key'] = self.api_key
            
            response = requests.get(self.api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                books = []
                for item in data.get('items', []):
                    volume_info = item.get('volumeInfo', {})
                    
                    book_data = {
                        'title': volume_info.get('title', 'No title'),
                        'authors': volume_info.get('authors', []),
                        'description': volume_info.get('description', '')[:500],
                        'published_date': volume_info.get('publishedDate', ''),
                        'page_count': volume_info.get('pageCount', 0),
                        'categories': volume_info.get('categories', []),
                        'preview_link': volume_info.get('previewLink', ''),
                        'info_link': volume_info.get('infoLink', '')
                    }
                    books.append(book_data)
                    
                    authors_str = ', '.join(book_data['authors'][:2])
                    logger.info(f"     {book_data['title'][:50]} - {authors_str}")
                
                logger.info(f"  Found {len(books)} books")
                logger.info(f"  Quota used: 1 request (1,000 daily limit)")
                logger.info(f"  Cost: $0")
                
                return {
                    'books': books,
                    'total_results': data.get('totalItems', 0),
                    'quota_used': 1,
                    'cost': 0
                }
            else:
                logger.warning(f"   Books API returned status {response.status_code}")
                return {'error': f'Status {response.status_code}', 'books': []}
                
        except Exception as e:
            logger.error(f"  Books search error: {e}")
            return {'error': str(e), 'books': []}


class GoogleGeminiConnector:
    """Google Gemini API     (  !)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.available = False
        
        if not self.api_key:
            logger.warning("   Gemini API key not provided")
            logger.info("   Get free key: https://makersuite.google.com/app/apikey")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.available = True
            logger.info("  Gemini 1.5 Flash ready (FREE 1,500 requests/day!)")
            logger.info("     This is HUGE - free LLM integration!")
        except ImportError:
            logger.warning("   google-generativeai not installed")
            logger.info("   Install: pip install google-generativeai")
        except Exception as e:
            logger.error(f"  Gemini API error: {e}")
    
    def generate_content(self, prompt: str) -> Dict[str, Any]:
        """
        Gemini        
        
        Args:
            prompt:     
        
        Returns:
                   
        """
        if not self.available:
            return {
                'error': 'Gemini API not available',
                'setup': 'Get free key at https://makersuite.google.com/app/apikey'
            }
        
        try:
            response = self.model.generate_content(prompt)
            
            logger.info("  Gemini response generated")
            logger.info(f"  Quota used: 1 request (1,500 daily limit)")
            logger.info(f"  Cost: $0")
            
            return {
                'text': response.text,
                'quota_used': 1,
                'cost': 0
            }
            
        except Exception as e:
            logger.error(f"  Gemini generation error: {e}")
            return {'error': str(e)}


# Demo
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print(" " * 20 + "  GOOGLE FREE SERVICES DEMO  ")
    print("=" * 80)
    print()
    print("        : '                         '")
    print("              '        '")
    print()
    print("               !")
    print()
    print("    YouTube Data API: 10,000 units/day (  !)")
    print("    Google Books API: 1,000 requests/day (  !)")
    print("    Gemini 1.5 Flash: 1,500 requests/day (  !)")
    print()
    print("    Total cost: $0")
    print()
    print("=" * 80)
    print()
    
    # API     
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('YOUTUBE_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("    API              .")
        print()
        print("   API        :")
        print("1. https://console.cloud.google.com   ")
        print("2.         (  !)")
        print("3. YouTube Data API v3    ")
        print("4.             API     ")
        print("5. .env     GOOGLE_API_KEY=your_key   ")
        print()
        print("Gemini API   (  ,          !):")
        print("1. https://makersuite.google.com/app/apikey   ")
        print("2. API      (  !)")
        print("3. .env     GEMINI_API_KEY=your_key   ")
        print()
        print("=" * 80)
        sys.exit(0)
    
    #        
    connector = GoogleFreeServicesConnector(api_key)
    
    #      
    topic = "artificial intelligence"
    
    print(f"  Learning topic: {topic}")
    print()
    
    #            !
    services = ['youtube', 'books']
    if gemini_key:
        services.append('gemini')
    
    results = connector.learn_topic_with_google(topic, use_services=services)
    
    print()
    print("=" * 80)
    print("  Results Summary:")
    print("=" * 80)
    print(f"Topic: {results['topic']}")
    print(f"Services used: {', '.join(results['services_used'])}")
    print()
    
    if 'youtube' in results['data']:
        yt = results['data']['youtube']
        if 'videos' in yt:
            print(f"  YouTube: {len(yt['videos'])} videos found")
    
    if 'books' in results['data']:
        books = results['data']['books']
        if 'books' in books:
            print(f"  Books: {len(books['books'])} books found")
    
    if 'gemini' in results['data']:
        gemini = results['data']['gemini']
        if 'text' in gemini:
            print(f"  Gemini: Enhanced with LLM!")
    
    print()
    print(f"  Total cost: $0")
    print()
    print("               :")
    print("   '                         !'")
    print()
    print("=" * 80)
