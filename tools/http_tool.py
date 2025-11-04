from typing import Dict, Any
import requests
from bs4 import BeautifulSoup


def fetch_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetches a URL and extracts a simple summary: status, title, and plain text up to a limit.
    """
    try:
        r = requests.get(url, timeout=timeout)
        out: Dict[str, Any] = {
            'status_code': r.status_code,
            'headers': dict(r.headers),
            'url': r.url,
        }
        ctype = r.headers.get('content-type', '')
        if 'text/html' in ctype:
            soup = BeautifulSoup(r.text, 'html.parser')
            title = soup.title.string.strip() if (soup.title and soup.title.string) else ''
            # crude text extraction
            text = soup.get_text(" ")
            if len(text) > 5000:
                text = text[:5000]
                out['truncated'] = True
            out.update({'title': title, 'text': text})
        else:
            # Non-HTML: return short body preview
            body = r.text
            if len(body) > 2000:
                body = body[:2000]
                out['truncated'] = True
            out['body'] = body
        return out
    except Exception as e:
        return {'error': str(e)}

