"""
TruthRadar - News Article Web Scraper
-------------------------------------
Extracts news article text, headline, author, publication date,
and metadata from web URLs using requests and BeautifulSoup.

Follows ethical scraping practices, handles errors gracefully,
and never bypasses paywalls, CAPTCHAs, or authentication.
"""

from typing import Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import config
from src.utils import clean_url, logger


def scrape_article_from_url(url: str, timeout: int = config.MAX_SCRAPER_TIMEOUT) -> Dict[str, Any]:
    """
    Fetches and parses a news article from a given web URL.

    Args:
        url: Web URL string.
        timeout: Network request timeout in seconds.

    Returns:
        Dict containing:
            - success: bool
            - title: str
            - text: str
            - author: Optional[str]
            - publish_date: Optional[str]
            - domain: str
            - url: str
            - error: Optional[str]
    """
    if not url or not url.strip():
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": None,
            "publish_date": None,
            "domain": "",
            "url": url,
            "error": "URL cannot be empty."
        }

    formatted_url = clean_url(url)
    domain = urlparse(formatted_url).netloc

    headers = {
        "User-Agent": config.DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        response = requests.get(formatted_url, headers=headers, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 403 or response.status_code == 401:
            return {
                "success": False,
                "title": "",
                "text": "",
                "author": None,
                "publish_date": None,
                "domain": domain,
                "url": formatted_url,
                "error": f"Access restricted by the website (HTTP {response.status_code}). Please copy and paste the article text manually."
            }
        
        if response.status_code != 200:
            return {
                "success": False,
                "title": "",
                "text": "",
                "author": None,
                "publish_date": None,
                "domain": domain,
                "url": formatted_url,
                "error": f"Failed to retrieve web page (HTTP status code {response.status_code}). Please verify the URL."
            }

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # 1. Extract Title / Headline
        title = ""
        # Check OpenGraph or Twitter meta tags first
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        elif soup.title:
            title = soup.title.get_text().strip()

        # 2. Extract Author
        author = None
        author_meta = (
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", property="article:author")
            or soup.find(class_=lambda x: x and any(a in x.lower() for a in ["author-name", "byline", "author"]))
        )
        if author_meta:
            if author_meta.name == "meta" and author_meta.get("content"):
                author = author_meta["content"].strip()
            elif hasattr(author_meta, "get_text"):
                author_text = author_meta.get_text().strip()
                if 2 < len(author_text) < 60:
                    author = author_text

        # 3. Extract Date
        pub_date = None
        date_meta = (
            soup.find("meta", property="article:published_time")
            or soup.find("meta", attrs={"name": "publication_date"})
            or soup.find("meta", attrs={"name": "date"})
            or soup.find("time")
        )
        if date_meta:
            if date_meta.name == "meta" and date_meta.get("content"):
                pub_date = date_meta["content"][:10].strip()
            elif date_meta.name == "time" and date_meta.get("datetime"):
                pub_date = date_meta["datetime"][:10].strip()
            elif hasattr(date_meta, "get_text"):
                dt = date_meta.get_text().strip()
                if 4 < len(dt) < 40:
                    pub_date = dt

        # 4. Extract Article Body Text
        # Remove script, style, nav, footer, ads, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "svg"]):
            tag.decompose()

        paragraphs = []
        # Check for standard semantic article container
        article_tag = soup.find("article") or soup.find("div", class_=lambda x: x and any(c in x.lower() for c in ["article-body", "story-body", "post-content", "entry-content", "content__body"]))
        
        target_container = article_tag if article_tag else soup

        for p in target_container.find_all("p"):
            p_text = p.get_text().strip()
            # Filter out tiny boilerplate strings, cookie notices, social share links
            if len(p_text) > 40 and not any(kw in p_text.lower() for kw in ["all rights reserved", "subscribe now", "sign up for", "cookie policy", "terms of service"]):
                paragraphs.append(p_text)

        article_text = "\n\n".join(paragraphs)

        if not article_text or len(article_text) < 50:
            return {
                "success": False,
                "title": title,
                "text": "",
                "author": author,
                "publish_date": pub_date,
                "domain": domain,
                "url": formatted_url,
                "error": "Could not extract substantive article text. The page might rely heavily on JavaScript. Please copy and paste the article text manually."
            }

        return {
            "success": True,
            "title": title,
            "text": article_text,
            "author": author,
            "publish_date": pub_date,
            "domain": domain,
            "url": formatted_url,
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": None,
            "publish_date": None,
            "domain": domain,
            "url": formatted_url,
            "error": f"Request timed out after {timeout} seconds. The web server took too long to respond."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": None,
            "publish_date": None,
            "domain": domain,
            "url": formatted_url,
            "error": "Failed to establish a network connection. Please check your internet connection or the URL."
        }
    except Exception as e:
        logger.error(f"Scraper error: {e}", exc_info=True)
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": None,
            "publish_date": None,
            "domain": domain,
            "url": formatted_url,
            "error": f"Failed to extract content: {e}. Please copy and paste the article text manually."
        }
