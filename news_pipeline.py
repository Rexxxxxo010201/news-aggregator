"""Imports and Constants"""

iimport json
import re
import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd

import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))


# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Constants
API_KEY = "257f80667e2545498f541dc0f79ab667"
BASE_URL = "https://newsapi.org/v2/top-headlines"
VALID_CATEGORIES = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
DEFAULT_COUNTRY = "us"


"""Data Model for News Article"""

@dataclass
class NewsArticle:
    title: str
    description: Optional[str]
    url: str
    source_name: str
    published_at: str
    category: Optional[str] = None  
    def __str__(self):
        return (
            f"{self.title}\n"
            f"Source: {self.source_name} | Published: {self.published_at}\n"
            f"{self.description or 'No description'}\n"
            f"URL: {self.url}\n"
        )

"""NewsFetcher Class"""

class NewsFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.cache = {}  # Caches responses based on query params

 
    def get_news(
        self,
        category: Optional[str] = None,
        country: str = "us",
        source: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 10
    ) -> List['NewsArticle']:

        # Create a unique key for the current query
        cache_key = (category, country, source, query, page_size)

        if cache_key in self.cache:
            logging.info("✅ Using cached API response")
            return self.cache[cache_key]

        # Construct query params
        params = {
            "apiKey": self.api_key,
            "country": country,
            "pageSize": page_size
        }
        if category:
            params["category"] = category
        if source:
            params["sources"] = source
        if query:
            params["q"] = query

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "articles" not in data or not isinstance(data["articles"], list):
                logging.warning("Unexpected response format from NewsAPI.")
                return []

            articles = self._parse_articles(data["articles"], category)

            # Cache the result
            self.cache[cache_key] = articles
            return articles

        except requests.exceptions.RequestException as e:
            logging.error(f"Network/API error while fetching news: {e}")
            return []

        except Exception as e:
            logging.error(f"Unexpected error in get_news(): {e}")
            return []



    def _parse_articles(self, articles_json: List[dict], category: Optional[str]) -> List[NewsArticle]:

        articles = []
        for item in articles_json:
            article = NewsArticle(
                title=item.get("title", "No title"),
                description=item.get("description"),
                url=item.get("url", ""),
                source_name=item.get("source", {}).get("name", "Unknown"),
                published_at=item.get("publishedAt"),  # remove self._format_date()
                category=category
            )
            articles.append(article)
        return articles



"""CLI Interaction"""

def run_cli():
    fetcher = NewsFetcher(API_KEY)
    print("🌐 Welcome to the News Aggregator!")
    print("Available categories:", ", ".join(VALID_CATEGORIES))
    print("Example country codes: au (Australia), us (USA), in (India), gb (UK), ca (Canada), etc.\n")

    # Ask for country code
    country = input("Enter 2-letter country code [default: au]: ").strip().lower()

    if not country:
        country = "au"

    # Ask for category
    category = input("Enter news category (leave blank to skip): ").strip().lower()
    if isinstance(category, list):
      category = category[0]

    if category and category not in VALID_CATEGORIES:
        print(f"⚠️ Invalid category '{category}' — defaulting to general.")
        category = "general"

    # keyword search
    keyword = input("Enter keyword to search in articles (optional): ").strip()

    try:
        news_list = fetcher.get_news(
            country=country,
            category=category if category else None,
            query=keyword if keyword else None,
            page_size=5
        )
        if not news_list:
            print("\n😕 No articles found for this selection.")
        else:
            print("\n🗞️ Top News Headlines:\n")
            for i, article in enumerate(news_list, start=1):
                print(f"{i}. {article}\n")

    except Exception as e:
        print(f"❌ Error fetching news: {e}")

# run_cli()



"""NewsScraper Class with BeautifulSoup"""



class NewsScraper:

    robots_cache = {}
    scrape_cache = {}

    def __init__(self, url: str):
        self.url = url
        self.soup = None



    def fetch_html(self) -> None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/"
        }

        session = requests.Session()

        retries = Retry(
            total=1,                # number of retry attempts
            backoff_factor=1,       # wait time: 1s, then 2s, then 4s
            status_forcelist=[429, 500, 502, 503, 504],  # retry on these
            raise_on_status=False   # don't raise exception, just retry
        )

        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.mount("http://", HTTPAdapter(max_retries=retries))

        try:
            response = session.get(self.url, headers=headers, timeout=10)
            response.encoding = "utf-8"
            response.raise_for_status()
            self.soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            raise Exception(f"❌ Failed to fetch article HTML: {e}")





    def extract_main_text(self) -> Optional[str]:
        if not self.soup:
            return None

        # Try JSON-LD blocks that may contain articleBody
        json_ld_blocks = self.soup.find_all("script", type="application/ld+json")
        for block in json_ld_blocks:
            try:
                data = json.loads(block.string)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "articleBody" in item:
                            return item["articleBody"]
                elif isinstance(data, dict) and "articleBody" in data:
                    return data["articleBody"]
            except Exception as e:
                logging.warning(f"Failed to parse JSON-LD block: {e}")
                continue

        # Fallback to div/article scraping if needed
        return None



    def extract_author(self) -> Optional[str]:
        if not self.soup:
            return None
        # Check meta tags
        meta_author = self.soup.find("meta", {"name": "author"}) or self.soup.find("meta", {"property": "article:author"})
        if meta_author and meta_author.get("content"):
            return meta_author["content"]
        # Fallback from byline
        byline = self.soup.find("span", class_="byline") or self.soup.find("p", class_="author")
        if byline:
            return byline.get_text(strip=True)
        return None


    def extract_tags(self) -> Optional[List[str]]:
        if not self.soup:
            return None

        tag_candidates = self.soup.select(".tag, .keywords, .topics, .article-tags a, a[rel='tag']")
        tags = [tag.get_text(strip=True) for tag in tag_candidates if tag.get_text(strip=True)]

        return tags if tags else None



    def scrape_all(self) -> dict:

        if self.url in self.scrape_cache:
            logging.info(f"✅ Using cached scrape result for: {self.url}")
            return self.scrape_cache[self.url]

        try:
            self.fetch_html()
            json_ld_blocks = self.soup.find_all("script", type="application/ld+json")

            content = None
            author = None
            tags = []

            for block in json_ld_blocks:
                try:
                    data = json.loads(block.string)

                    # Sometimes it's a list of schema entries
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                content = content or item.get("articleBody")
                                if isinstance(content, list):
                                     # join paragraphs into one big blob
                                    content = " ".join(content)
                                author = author or self.extract_author_name(item.get("author"))
                    elif isinstance(data, dict):
                        content = content or data.get("articleBody")
                        author = author or self.extract_author_name(data.get("author"))

                except Exception as e:
                    logging.warning(f"⚠️ Failed to parse JSON-LD block: {e}")
                    continue

            # Fallback to newspaper3k if no content
            if not content:
                try:
                    article = Article(self.url)
                    article.download()
                    article.parse()
                    content = article.text or None
                    author = author or article.authors[0] if article.authors else None
                except Exception as e:
                    logging.warning(f"📄 newspaper3k fallback failed: {e}")

                # 🔁 Check robots.txt now, if content is still missing
                if not content:
                    if not self.is_scraping_allowed(self.url):
                        domain = urlparse(self.url).netloc
                        logging.warning(f"⚠️ Skipping {domain} — scraping disallowed by robots.txt.")
                        return {
                            "content": "⚠️ This article is blocked by site policy (robots.txt).",
                            "author": "Unavailable",
                            "tags": []
                        }
            self.scrape_cache[self.url] = {
                "content": content or "Content not available",
                "author": author or "Unknown",
                "tags": tags
            }
            return {
                "content": content or "Content not available",
                "author": author or "Unknown",
                "tags": tags
            }


        except Exception as e:
            logging.error(f"❌ Scraping failed: {e}")

            if not self.is_scraping_allowed(self.url):
                domain = urlparse(self.url).netloc
                logging.warning(f"⚠️ Skipping {domain} — scraping disallowed by robots.txt.")
                return {
                    "content": "⚠️ This article is blocked by site policy (robots.txt).",
                    "author": "Unavailable",
                    "tags": []
                }

            return {
                "content": "Content not available",
                "author": "Unknown",
                "tags": []
            }






    @staticmethod
    def extract_author_name(author_field):
        if not author_field:
            return None

        if isinstance(author_field, dict):
            return author_field.get("name") or NewsScraper.extract_author_from_url(author_field.get("url"))
        elif isinstance(author_field, str):
            return NewsScraper.extract_author_from_url(author_field) if author_field.startswith("http") else author_field
        elif isinstance(author_field, list):
            names = [NewsScraper.extract_author_name(a) for a in author_field if a]
            return ", ".join(filter(None, names))
        return None

    @staticmethod
    def extract_author_from_url(url):
        if not url:
            return None
        try:
            name = url.rstrip("/").split("/")[-1].replace("-", " ")
            return name.title()
        except Exception:
            return None

    @classmethod
    def is_scraping_allowed(cls, url: str) -> bool:
        domain = urlparse(url).scheme + "://" + urlparse(url).netloc
        if domain in cls.robots_cache:
            return cls.robots_cache[domain]

        try:
            robots_url = domain + "/robots.txt"
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                content = response.text.lower()
                if "user-agent: *" in content and "disallow: /" in content:
                    cls.robots_cache[domain] = False
                    return False
        except Exception as e:
            logging.warning(f"Could not check robots.txt for {domain}: {e}")

        cls.robots_cache[domain] = True
        return True

"""EnrichedNewsArticle — Extend Original"""

@dataclass
class EnrichedNewsArticle(NewsArticle):
    content: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)

"""Example: Enrich One Article"""

def run_scrape(news_article: NewsArticle) -> EnrichedNewsArticle:
    print(f"🔍 Scraping: {news_article.title}\nURL: {news_article.url}")
    scraper = NewsScraper(news_article.url)
    extra_data = scraper.scrape_all()

    enriched = EnrichedNewsArticle(
        title=news_article.title,
        description=news_article.description,
        url=news_article.url,
        source_name=news_article.source_name,
        published_at=news_article.published_at,
        content=extra_data["content"],
        author=extra_data["author"],
        tags=extra_data["tags"]
    )

    return enriched

fetcher = NewsFetcher(API_KEY)
articles = fetcher.get_news(category="business", country="us", page_size=1)

if not articles:
    print("⚠️ No articles found. Try a different category, country, or check your API limits.")
else:
    article = articles[0]
    enriched = run_scrape(article)

    print("📰", enriched.title)
    print("📄", (enriched.content or "No content available")[:1000])  # Print a preview
    print("✍️", enriched.author or "Unknown")
    print("🏷️", enriched.tags or [])






class NewsPipeline:
    def __init__(self, fetcher):
        self.fetcher = fetcher

# Combines API + Scraped Data
    def collect_data(self, articles: List[NewsArticle]) -> List[Dict]:
        combined_records = []

        for i, article in enumerate(articles, 1):
            print(f"🔄 [{i}/{len(articles)}] Processing: {article.title}")

            try:
                enriched = run_scrape(article)

                combined = {
                    "title": article.title,
                    "summary": article.description or "N/A",
                    "url": article.url,
                    "source": article.source_name,
                    "published_date": article.published_at,
                    "author": enriched.author or "Unknown",
                    "content": enriched.content or "Content not available",
                    "category": getattr(article, "category", "Unknown"),
                    "tags": enriched.tags or []
                }

                combined_records.append(combined)

            except Exception as e:
                logging.warning(f"⚠️ Failed to process article '{article.title}': {e}")
                continue

        return combined_records


# Clean and Deduplicate Records
    def clean_data(self, records: List[Dict]) -> List[Dict]:
        seen_urls = set()
        cleaned = []
        for record in records:
            # Ensure all fields are strings (handle lists or other types)
            record["title"] = " ".join(record["title"]) if isinstance(record["title"], list) else str(record["title"])
            record["summary"] = " ".join(record["summary"]) if isinstance(record["summary"], list) else str(record["summary"])
            record["author"] = " ".join(record["author"]) if isinstance(record["author"], list) else str(record["author"])
            record["source"] = " ".join(record["source"]) if isinstance(record["source"], list) else str(record["source"])
            
            # Normalize and clean content
            content = record["content"]
            if isinstance(content, list):
                content = " ".join(content)
            content = str(content)
            content = re.sub(r"\s+", " ", content).strip()
            content = (
                content.replace("’", "'")
                    .replace("“", '"')
                    .replace("”", '"')
                    .replace("–", "-")
                    .replace("—", "-")
            )
            content = unicodedata.normalize("NFKD", content).encode("ascii", "ignore").decode()
            record["content"] = content

            # Normalize tags
            record["tags"] = [str(tag).strip() for tag in record.get("tags", []) if tag]

            # Filter out junky articles
            if len(record["content"]) < 100 or len(record["title"]) < 5:
                continue

            cleaned.append(record)


        return cleaned



# Normalize Schema & Fill Gaps
    def format_date(self, date_str: str) -> str:
        try:
            # Remove 'Z' and parse as UTC
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).isoformat()
        except Exception:
            return "Unknown"

    def format_data(self, records: List[Dict]) -> List[Dict]:
        formatted = []

        for record in records:
            formatted_record = {
                "title": record.get("title", "N/A"),
                "category": record.get("category", "Unknown"),
                "summary": record.get("summary", "N/A"),
                "url": record.get("url", "N/A"),
                "source": record.get("source", "Unknown"),
                "author": record.get("author", "Unknown"),
                "content": record.get("content", "Content not available"),
                "keywords": self.extract_keywords_from_content(record.get("content", "")),

                "published_date": self.format_date(record.get("published_date")),
                "content_preview": record.get("content", "")[:300] + "..."  # Preview
            }

            formatted.append(formatted_record)

        return formatted



# Return a Pandas DataFrame
    def export_data(self, records: List[Dict]) -> pd.DataFrame:
        if not records:
            print("⚠️ No data to export.")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # re-order columns for consistency
        column_order = [
            "title", "category", "summary", "author", "source", "published_date",
            "content_preview", "keywords", "url", "content"
        ]
        df = df[[col for col in column_order if col in df.columns]]

        return df


    def process_articles(self, category="general", country="us", page_size=5) -> pd.DataFrame:


        articles = self.fetcher.get_news(category=category, country=country, page_size=page_size)

        # Attach category to each article object
        for article in articles:
            # article.category = category
            if isinstance(category, list):
                article.category = category[0]
            else:
                article.category = category
        articles = self.fetcher.get_news(category=category, country=country, page_size=page_size)
        combined = self.collect_data(articles)
        cleaned = self.clean_data(combined)
        formatted = self.format_data(cleaned)
        df = self.export_data(formatted)
        return df

    def extract_keywords_from_content(self, text: str, top_n: int = 5) -> List[str]:
        if not text:
            return []

        # Tokenize and clean
        words = re.findall(r"\b\w+\b", text.lower())
        filtered = [word for word in words if word not in STOPWORDS and len(word) > 3]

        # Count and return top-N
        common = Counter(filtered).most_common(top_n)
        return [word for word, _ in common]



# Valid categories from NewsAPI
VALID_CATEGORIES = [
    "business", "entertainment", "general",
    "health", "science", "sports", "technology"
]

# Instantiate fetcher and pipeline
fetcher = NewsFetcher(API_KEY)
pipeline = NewsPipeline(fetcher)

# Loop over all categories and collect data
final_dfs = []

for category in VALID_CATEGORIES:
    print(f"\n🔎 Fetching articles in category: {category}")

    try:
        df_temp = pipeline.process_articles(category=category, country="us", page_size=100)

        if df_temp.empty:
            print(f"⚠️ No articles returned for category '{category}'")
            continue

        final_dfs.append(df_temp)

    except Exception as e:
        print(f"❌ Failed to fetch or process category '{category}': {e}")
        continue

# Combine all category results into a single DataFrame
if final_dfs:
    df = pd.concat(final_dfs, ignore_index=True)
    df.drop_duplicates(subset="url", inplace=True)  # Deduplicate by article URL
else:
    df = pd.DataFrame()



# Inspect final dataset
print("\n✅ Final DataFrame built")
print("🧮 Shape:", df.shape)
print("📋 Columns:", df.columns.tolist())
print("🔍 Sample preview:")
print(df.head())
print(df[['title', 'category']].head())


df.to_csv("news_dataset.csv", index=False)