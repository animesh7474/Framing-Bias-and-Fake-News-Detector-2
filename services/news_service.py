"""
news_service.py — Service layer for non-blocking news context retrieval.
Domain: Computer Networks + Big Data Analytics
"""

import re
import urllib.parse
import feedparser
from logger import get_logger

log = get_logger("news_service")

def fetch_related_news(text: str, max_results: int = 5):
    """
    Fetches live news context via Google News RSS synchronously.
    Extremely stable, no compiled dependencies like curl_cffi.
    """
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    query = " ".join(clean_text.split())[:120]
    
    log.info(f"News Search started for query: '{query[:60]}...'")
    
    try:
        encoded_query = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(rss_url)
        results = []
        
        for entry in feed.entries[:max_results]:
            results.append({
                "title": entry.get("title", "No Title"),
                "source": entry.get("source", {}).get("title", "Google News"),
                "date": entry.get("published", "Recent"),
                "link": entry.get("link", ""),
                "snippet": entry.get("title", "No description available.")  # Google RSS puts most info in title
            })
            
        log.info(f"Final news context count: {len(results)}")
        return results
    except Exception as e:
        log.error(f"News Service Final Error: {e}")
        return []
