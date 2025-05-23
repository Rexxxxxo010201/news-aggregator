import unittest
from news_pipeline import NewsFetcher, NewsScraper, NewsPipeline, NewsArticle


API_KEY =  "257f80667e2545498f541dc0f79ab667"

class TestNewsFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = NewsFetcher(API_KEY)

    def test_fetch_count(self):
        articles = self.fetcher.get_news(category="technology", page_size=3)
        self.assertEqual(len(articles), 3)

    def test_article_object(self):
        articles = self.fetcher.get_news(category="health", page_size=1)
        self.assertIsInstance(articles[0], NewsArticle)
        self.assertTrue(articles[0].title)

class TestNewsPipeline(unittest.TestCase):
    def setUp(self):
        self.fetcher = NewsFetcher(API_KEY)
        self.pipeline = NewsPipeline(self.fetcher)

    def test_pipeline_output_dataframe(self):
        df = self.pipeline.process_articles(category="science", page_size=3)
        self.assertFalse(df.empty)
        self.assertIn("keywords", df.columns)
        self.assertEqual(len(df), 3)

    def test_keywords_format(self):
        df = self.pipeline.process_articles(category="sports", page_size=2)
        for kw in df["keywords"]:
            self.assertIsInstance(kw, list)
            self.assertLessEqual(len(kw), 5)

class TestNewsScraper(unittest.TestCase):
    def test_scraper_handles_robots_txt(self):
        # This URL is known to disallow scraping
        url = "https://www.washingtonpost.com/nation/2025/05/09/newark-airport-radar-outage/"
        scraper = NewsScraper(url)
        result = scraper.scrape_all()
        self.assertIn("blocked by site policy", result["content"])

print("🧪 Running test file...")
if __name__ == '__main__':
    unittest.main()
