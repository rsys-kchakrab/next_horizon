# FILE: build_training_dataset/build_training_database.py

"""
Training Content Web Scraper - Similar to JD Database Builder

This script scrapes training courses from major platforms like Coursera, Udemy, edX, etc.
and builds a comprehensive training_database.csv database.

Features:
- Searches multiple course platforms for each skill
- Extracts course metadata (title, description, provider, duration, price, etc.)
- Parallel processing for efficient data collection
- Robust error handling and retry mechanisms
- Supports both SerpAPI and DuckDuckGo search

Install:
    pip install ddgs beautifulsoup4 tldextract html5lib requests pandas certifi

Usage:
    python build_training_database.py --skills skill_list.csv --out training_database.csv --per_skill 10
"""

from __future__ import annotations
import argparse, csv, os, re, time, uuid, tldextract, html, logging, sys
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd

try:
    import certifi
    CERTIFI_PATH = certifi.where()
except Exception:
    CERTIFI_PATH = None

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# Major course platforms to prioritize
COURSE_PLATFORMS = [
    "coursera.org", "udemy.com", "edx.org", "udacity.com", "pluralsight.com",
    "linkedin.com/learning", "skillshare.com", "codecademy.com", "freecodecamp.org",
    "khanacademy.org", "masterclass.com", "futurelearn.com", "domestika.com",
    "brilliant.org", "datacamp.com", "kaggle.com/learn"
]

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def _import_ddg():
    """Import DuckDuckGo search library"""
    try:
        from ddgs import DDGS
        return "ddgs", DDGS
    except Exception:
        pass
    try:
        from duckduckgo_search import DDGS
        return "duckduckgo_search", DDGS
    except Exception:
        return None, None

def search_ddg(query: str, k: int = 15) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo"""
    pkg, DDGS = _import_ddg()
    if DDGS is None:
        logging.warning("ddgs/duckduckgo_search not installed; ddg search disabled.")
        return []
    
    results = []
    try:
        with DDGS() as ddgs:
            kwargs = dict(max_results=int(k))
            try:
                it = ddgs.text(query, **kwargs)
            except TypeError:
                it = ddgs.text(keywords=query, **kwargs)
            for x in it:
                results.append({
                    "title": (x.get("title") or ""),
                    "link": (x.get("href") or x.get("url") or ""),
                    "snippet": (x.get("body") or ""),
                    "source": pkg
                })
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.warning(f"ddg search error: {e}")
        return []
    return results

def search_serpapi(query: str, k: int = 15) -> List[Dict[str, Any]]:
    """Search using SerpAPI (Google Search)"""
    out = []
    if not SERPAPI_KEY:
        return out
    
    try:
        params = {"engine": "google", "q": query, "num": int(k), "api_key": SERPAPI_KEY}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        for it in (data.get("organic_results") or [])[:k]:
            out.append({
                "title": it.get("title", ""),
                "link": it.get("link", ""),
                "snippet": it.get("snippet", ""),
                "source": "serpapi"
            })
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.warning(f"serpapi search error: {e}")
    return out

def search_courses(query: str, engine: str = "auto", k: int = 15) -> List[Dict[str, Any]]:
    """Search for courses using specified engine"""
    if engine == "serpapi" or (engine == "auto" and SERPAPI_KEY):
        return search_serpapi(query, k)
    else:
        return search_ddg(query, k)

def create_session(proxy: Optional[str] = None, verify_tls: bool = True) -> requests.Session:
    """Create HTTP session with retries and configuration"""
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    
    s.verify = CERTIFI_PATH if verify_tls and CERTIFI_PATH else verify_tls
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    
    return s

def extract_course_metadata(url: str, html_content: str, platform: str) -> Dict[str, Any]:
    """Extract course metadata from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = {}
    
    try:
        # Extract title
        title_selectors = [
            'h1', '.course-title', '.title', '[data-testid="course-title"]',
            '.course-header h1', '.course-name', '.hero-title'
        ]
        title = None
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                title = element.get_text(strip=True)
                break
        metadata['title'] = title or "Course Title Not Found"
        
        # Extract description
        desc_selectors = [
            '.course-description', '.description', '.about-course',
            '[data-testid="course-description"]', '.course-overview',
            '.course-summary', '.syllabus-overview'
        ]
        description = None
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                desc_text = element.get_text(strip=True)
                if len(desc_text) > 50:  # Ensure meaningful description
                    description = desc_text[:500] + "..." if len(desc_text) > 500 else desc_text
                    break
        metadata['description'] = description or "Description not available"
        
        # Platform-specific metadata extraction
        if 'coursera.org' in platform:
            metadata.update(_extract_coursera_metadata(soup))
        elif 'udemy.com' in platform:
            metadata.update(_extract_udemy_metadata(soup))
        elif 'edx.org' in platform:
            metadata.update(_extract_edx_metadata(soup))
        else:
            metadata.update(_extract_generic_metadata(soup))
            
    except Exception as e:
        logging.warning(f"Error extracting metadata from {platform}: {e}")
    
    return metadata

def _extract_coursera_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract Coursera-specific metadata"""
    metadata = {}
    
    # Duration
    duration_elem = soup.select_one('[data-testid="duration"]') or soup.select_one('.course-duration')
    if duration_elem:
        duration_text = duration_elem.get_text(strip=True)
        hours = _parse_duration(duration_text)
        metadata['hours'] = hours
    
    # Price
    price_elem = soup.select_one('.price') or soup.select_one('[data-testid="price"]')
    if price_elem:
        price_text = price_elem.get_text(strip=True).lower()
        metadata['price'] = _categorize_price(price_text)
    
    # Rating
    rating_elem = soup.select_one('.rating') or soup.select_one('[data-testid="rating"]')
    if rating_elem:
        rating_text = rating_elem.get_text(strip=True)
        metadata['rating'] = _parse_rating(rating_text)
    
    return metadata

def _extract_udemy_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract Udemy-specific metadata"""
    metadata = {}
    
    # Duration
    duration_elem = soup.select_one('[data-purpose="video-content-length"]') or soup.select_one('.course-content-length')
    if duration_elem:
        duration_text = duration_elem.get_text(strip=True)
        hours = _parse_duration(duration_text)
        metadata['hours'] = hours
    
    # Price
    price_elem = soup.select_one('.price') or soup.select_one('[data-purpose="course-price"]')
    if price_elem:
        price_text = price_elem.get_text(strip=True).lower()
        metadata['price'] = _categorize_price(price_text)
    
    return metadata

def _extract_edx_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract edX-specific metadata"""
    metadata = {}
    
    # Duration
    duration_elem = soup.select_one('.course-effort') or soup.select_one('.duration')
    if duration_elem:
        duration_text = duration_elem.get_text(strip=True)
        hours = _parse_duration(duration_text)
        metadata['hours'] = hours
    
    return metadata

def _extract_generic_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract generic metadata for unknown platforms"""
    metadata = {}
    
    # Look for common duration patterns
    duration_patterns = [
        r'(\d+)\s*hours?',
        r'(\d+)\s*hrs?',
        r'(\d+)\s*minutes?',
        r'(\d+)\s*weeks?',
        r'(\d+)\s*months?'
    ]
    
    text_content = soup.get_text()
    for pattern in duration_patterns:
        match = re.search(pattern, text_content, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 'hour' in match.group(0).lower() or 'hr' in match.group(0).lower():
                metadata['hours'] = value
            elif 'minute' in match.group(0).lower():
                metadata['hours'] = round(value / 60, 1)
            elif 'week' in match.group(0).lower():
                metadata['hours'] = value * 10  # Estimate 10 hours per week
            elif 'month' in match.group(0).lower():
                metadata['hours'] = value * 40  # Estimate 40 hours per month
            break
    
    return metadata

def _parse_duration(duration_text: str) -> Optional[float]:
    """Parse duration text to hours"""
    if not duration_text:
        return None
    
    duration_text = duration_text.lower()
    
    # Extract numbers and units
    patterns = [
        (r'(\d+\.?\d*)\s*hours?', 1),
        (r'(\d+\.?\d*)\s*hrs?', 1),
        (r'(\d+\.?\d*)\s*minutes?', 1/60),
        (r'(\d+\.?\d*)\s*mins?', 1/60),
        (r'(\d+\.?\d*)\s*weeks?', 10),  # Assume 10 hours per week
        (r'(\d+\.?\d*)\s*months?', 40),  # Assume 40 hours per month
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, duration_text)
        if match:
            return round(float(match.group(1)) * multiplier, 1)
    
    return None

def _categorize_price(price_text: str) -> str:
    """Categorize price into free/low/medium/high"""
    if any(word in price_text for word in ['free', 'gratis', '$0']):
        return 'free'
    elif any(word in price_text for word in ['$', '€', '£', 'usd', 'eur', 'gbp']):
        # Try to extract numeric price
        price_match = re.search(r'[\$€£]?(\d+(?:\.\d{2})?)', price_text)
        if price_match:
            price_num = float(price_match.group(1))
            if price_num < 50:
                return 'low'
            elif price_num < 200:
                return 'medium'
            else:
                return 'high'
    return 'unknown'

def _parse_rating(rating_text: str) -> Optional[float]:
    """Parse rating text to float"""
    if not rating_text:
        return None
    
    match = re.search(r'(\d+\.?\d*)', rating_text)
    if match:
        return float(match.group(1))
    return None

def fetch_course_content(url: str, session: requests.Session, timeout: float = 10) -> Optional[str]:
    """Fetch course page content"""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return None

def is_course_platform(url: str) -> bool:
    """Check if URL is from a known course platform"""
    domain = tldextract.extract(url).registered_domain.lower()
    return any(platform in domain for platform in COURSE_PLATFORMS)

def process_skill(skill_name: str, session: requests.Session, engine: str, per_skill: int, timeout: float) -> List[Dict[str, Any]]:
    """Process a single skill to find training courses"""
    logging.info(f"Processing skill: {skill_name}")
    
    # Create search queries for different course types
    queries = [
        f"{skill_name} course tutorial training",
        f"learn {skill_name} online course",
        f"{skill_name} certification training",
        f"site:coursera.org OR site:udemy.com OR site:edx.org {skill_name}",
    ]
    
    all_results = []
    seen_urls = set()
    
    for query in queries:
        if len(all_results) >= per_skill:
            break
            
        try:
            search_results = search_courses(query, engine, k=min(per_skill, 10))
            
            for result in search_results:
                url = result.get('link', '').strip()
                if not url or url in seen_urls:
                    continue
                
                # Prioritize course platforms
                if not is_course_platform(url):
                    continue
                
                seen_urls.add(url)
                
                # Fetch course content
                html_content = fetch_course_content(url, session, timeout)
                if not html_content:
                    continue
                
                # Extract platform name
                domain = tldextract.extract(url).registered_domain
                platform_name = domain.replace('.com', '').replace('.org', '').title()
                
                # Extract metadata
                metadata = extract_course_metadata(url, html_content, domain)
                
                course_data = {
                    'training_id': f"T{len(all_results) + 1:04d}",
                    'skill': skill_name,
                    'title': metadata.get('title', result.get('title', 'Unknown Course')),
                    'description': metadata.get('description', result.get('snippet', 'No description')),
                    'provider': platform_name,
                    'hours': metadata.get('hours'),
                    'price': metadata.get('price', 'unknown'),
                    'rating': metadata.get('rating'),
                    'link': url
                }
                
                all_results.append(course_data)
                
                if len(all_results) >= per_skill:
                    break
                    
                # Add small delay to be respectful
                time.sleep(0.5)
                
        except Exception as e:
            logging.error(f"Error processing query '{query}' for skill '{skill_name}': {e}")
    
    logging.info(f"Found {len(all_results)} courses for skill: {skill_name}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Build training content database from web scraping")
    parser.add_argument("--skills", required=True, help="CSV file with skill names")
    parser.add_argument("--out", default="training_database.csv", help="Output CSV file")
    parser.add_argument("--per_skill", type=int, default=10, help="Courses per skill")
    parser.add_argument("--engine", choices=["auto", "serpapi", "ddg"], default="auto", help="Search engine")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--timeout", type=float, default=10, help="Request timeout")
    parser.add_argument("--proxy", help="HTTP proxy (optional)")
    parser.add_argument("--no-tls-verify", action="store_true", help="Disable TLS verification")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load skills
    try:
        skills_df = pd.read_csv(args.skills)
        skills = skills_df['skill_name'].tolist()
        logging.info(f"Loaded {len(skills)} skills from {args.skills}")
    except Exception as e:
        logging.error(f"Failed to load skills from {args.skills}: {e}")
        return
    
    # Create session
    session = create_session(args.proxy, not args.no_tls_verify)
    
    all_courses = []
    
    # Process skills in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_skill = {
            executor.submit(process_skill, skill, session, args.engine, args.per_skill, args.timeout): skill
            for skill in skills
        }
        
        for future in as_completed(future_to_skill):
            skill = future_to_skill[future]
            try:
                courses = future.result()
                all_courses.extend(courses)
            except Exception as e:
                logging.error(f"Failed to process skill {skill}: {e}")
    
    # Save results
    if all_courses:
        df = pd.DataFrame(all_courses)
        df.to_csv(args.out, index=False)
        logging.info(f"Saved {len(all_courses)} courses to {args.out}")
        
        # Print summary
        print(f"\n=== Training Database Built ===")
        print(f"Total courses: {len(all_courses)}")
        print(f"Skills covered: {df['skill'].nunique()}")
        print(f"Providers: {df['provider'].nunique()}")
        print(f"Output file: {args.out}")
    else:
        logging.warning("No courses found!")

if __name__ == "__main__":
    main()
