"""
news_service.py — Service layer for non-blocking news context retrieval.
Domain: Computer Networks + Big Data Analytics
"""

import asyncio
import re
import time
from duckduckgo_search import DDGS
from logger import get_logger

log = get_logger("news_service")

async def fetch_related_news(text: str, max_results: int = 5):
    """
    Fetches live news context with robust cleaning, retries, and fallback to text search.
    """
    # 1. Clean query: Remove special chars that might confuse search engine
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    query = " ".join(clean_text.split())[:120] # Shorten slightly for better relevance
    
    log.info(f"News Search started for query: '{query[:60]}...'")
    
    async def _try_search(mode="news"):
        for attempt in range(2):
            try:
                # Use a new DDGS instance per attempt to refresh state
                with DDGS() as ddgs:
                    if mode == "news":
                        items = ddgs.news(query, safesearch="off", max_results=max_results)
                    else:
                        # Fallback to general text search if news fails (often less restricted)
                        items = ddgs.text(query, safesearch="off", max_results=max_results, timelimit="d")
                    
                    if items:
                        log.info(f"DDGS {mode} results found: {len(items)} (Attempt {attempt+1})")
                        return items
            except Exception as e:
                log.warning(f"DDGS {mode} Attempt {attempt+1} failed: {e}")
                if "403" in str(e) or "Ratelimit" in str(e):
                    await asyncio.sleep(1.5) # Wait before retry
                else:
                    break # Don't retry on other errors
        return []

    try:
        # Try News Search first
        results_raw = await _try_search(mode="news")
        
        # If News failed, fallback to Text search (with 'day' filter for recency)
        if not results_raw:
            log.info("News search returned zero results. Falling back to Text search...")
            results_raw = await _try_search(mode="text")

        results = []
        for r in results_raw:
            results.append({
                "title": r.get("title", "No Title"),
                "source": r.get("source") or r.get("href", "Unknown Source"),
                "date": r.get("date", "Recent"),
                "link": r.get("url") or r.get("href", ""),
                "snippet": r.get("body") or r.get("snippet", "No description available.")
            })
        
        log.info(f"Final news context count: {len(results)}")
        return results
    except Exception as e:
        log.error(f"News Service Final Error: {e}")
        return []
