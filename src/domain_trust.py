"""
src/domain_trust.py - Domain Credibility & Source Reputation Intelligence Engine
-------------------------------------------------------------------------------
Evaluates publication credibility using domain reputation databases, 
satire detection registries, and heuristic TLD risk profiling.
"""

from urllib.parse import urlparse
import re
from typing import Dict, Any

# Verified High-Credibility International & Regional News Domains
VERIFIED_NEWS_DOMAINS = {
    "reuters.com": "Reuters (Tier 1 International Wire Service)",
    "apnews.com": "Associated Press (Tier 1 Global News Agency)",
    "bbc.com": "BBC News (Public Broadcaster - High Factuality)",
    "bbc.co.uk": "BBC News UK",
    "nytimes.com": "The New York Times",
    "theguardian.com": "The Guardian",
    "washingtonpost.com": "The Washington Post",
    "wsj.com": "The Wall Street Journal",
    "bloomberg.com": "Bloomberg News",
    "npr.org": "National Public Radio (NPR)",
    "thehindu.com": "The Hindu (National Indian Broadsheet)",
    "indianexpress.com": "The Indian Express",
    "ndtv.com": "NDTV News",
    "hindustantimes.com": "Hindustan Times",
    "economictimes.indiatimes.com": "The Economic Times",
    "nature.com": "Nature Publishing Group (Peer-Reviewed Science)",
    "sciencedirect.com": "ScienceDirect (Academic Research)",
    "pnas.org": "Proceedings of the National Academy of Sciences",
    "politifact.com": "PolitiFact (Verified Fact-Checking Registry)",
    "snopes.com": "Snopes (Verified Fact-Checking Registry)",
    "factcheck.org": "FactCheck.org (Annenberg Public Policy Center)",
    "altnews.in": "Alt News (IFCN Certified Fact-Checker)"
}

# Recognized Satire & Parody Publications (Crucial distinction in NLP)
SATIRE_DOMAINS = {
    "theonion.com": "The Onion (Satirical Newspaper & Humor Site)",
    "babylonbee.com": "The Babylon Bee (Christian News Satire)",
    "clickhole.com": "ClickHole (Parody Clickbait Media)",
    "borowitzreport.com": "The Borowitz Report (Satire)",
    "fakingnews.com": "Faking News (Indian Satirical Outlet)",
    "thepoke.co.uk": "The Poke (British Satire)",
    "waterfordwhispersnews.com": "Waterford Whispers (Irish Satire)"
}

# Documented Disinformation & Low-Credibility Conspiracy Portals
FLAGGED_DECEPTIVE_DOMAINS = {
    "infowars.com": "InfoWars (Documented Conspiracy Platform)",
    "naturalnews.com": "NaturalNews (Pseudoscience & Health Disinformation)",
    "beforeitsnews.com": "Before It's News (Unverified User Submissions)",
    "worldnewsdailyreport.com": "World News Daily Report (Fabricated Hoaxes)",
    "newspunch.com": "NewsPunch (Documented Fake News Portal)",
    "thegatewaypundit.com": "The Gateway Pundit (Frequent Debunked Claims)"
}

# Suspicious / High-Abuse Top-Level Domains (Spamhaus & ICANN research)
SUSPICIOUS_TLDS = {".xyz", ".top", ".buzz", ".club", ".click", ".country", ".kim", ".stream", ".loan"}

def evaluate_domain_trust(url_or_domain: str) -> Dict[str, Any]:
    """
    Evaluates domain reputation and outputs a trust rating with academic contextual notes.
    """
    if not url_or_domain or not url_or_domain.strip():
        return {
            "domain": "Unknown",
            "trust_tier": "Unspecified",
            "trust_score": 50,
            "is_satire": False,
            "badge_color": "gray",
            "explanation": "No valid URL provided."
        }

    raw = url_or_domain.strip().lower()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
        netloc = parsed.netloc.lower()
        # Strip port and www. prefix
        netloc = re.sub(r':\d+$', '', netloc)
        clean_domain = re.sub(r'^www\.', '', netloc)
    except Exception:
        clean_domain = "unknown"

    # 1. Check Satire
    for satire_domain, desc in SATIRE_DOMAINS.items():
        if clean_domain == satire_domain or clean_domain.endswith("." + satire_domain):
            return {
                "domain": clean_domain,
                "trust_tier": "Satire / Parody",
                "trust_score": 25,
                "is_satire": True,
                "badge_color": "orange",
                "title": desc,
                "explanation": f"Domain is an acknowledged satire publication ({desc}). The content is intentionally humorous/ironic, not factual news."
            }

    # 2. Check Verified Mainstream
    for verified_domain, desc in VERIFIED_NEWS_DOMAINS.items():
        if clean_domain == verified_domain or clean_domain.endswith("." + verified_domain):
            return {
                "domain": clean_domain,
                "trust_tier": "High Trust / Verified Journalism",
                "trust_score": 92,
                "is_satire": False,
                "badge_color": "green",
                "title": desc,
                "explanation": f"Established publication with formal editorial accountability: {desc}."
            }

    # 3. Check Flagged Disinformation
    for flagged_domain, desc in FLAGGED_DECEPTIVE_DOMAINS.items():
        if clean_domain == flagged_domain or clean_domain.endswith("." + flagged_domain):
            return {
                "domain": clean_domain,
                "trust_tier": "High Risk / Documented Misinformation",
                "trust_score": 10,
                "is_satire": False,
                "badge_color": "red",
                "title": desc,
                "explanation": f"Domain has a documented history of spreading debunked conspiracies and hoaxes ({desc})."
            }

    # 4. Check Suspicious TLDs
    for tld in SUSPICIOUS_TLDS:
        if clean_domain.endswith(tld):
            return {
                "domain": clean_domain,
                "trust_tier": "Unverified / High-Risk TLD",
                "trust_score": 35,
                "is_satire": False,
                "badge_color": "yellow",
                "title": f"Domain using `{tld}` TLD",
                "explanation": f"Domain uses `{tld}`, an uncurated top-level domain frequently associated with throwaway clickbait and scrapers."
            }

    # 5. Generic Independent / Unlisted Domain
    return {
        "domain": clean_domain,
        "trust_tier": "Independent / Unrated Domain",
        "trust_score": 60,
        "is_satire": False,
        "badge_color": "blue",
        "title": "Independent Online Source",
        "explanation": "Domain is not in our known whitelist or blacklist registries. Linguistic NLP analysis should serve as the primary verification signal."
    }
