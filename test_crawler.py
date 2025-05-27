from persian_crawler import PersianWebCrawler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_crawler():
    # Test with a small number of pages first
    crawler = PersianWebCrawler(
        base_url='https://udrc.ir',
        max_pages=3  # Start with a small number for testing
    )
    
    print("Starting test crawl...")
    crawler.crawl()
    
    # Print summary
    print("\nCrawl completed!")
    print(f"Pages crawled: {crawler.pages_crawled}")
    print(f"Links found: {len(crawler.visited_urls)}")
    
    # Print the first few links as a sample
    print("\nSample of crawled pages:")
    for i, url in enumerate(list(crawler.visited_urls)[:3]):
        print(f"{i+1}. {url}")

if __name__ == "__main__":
    test_crawler()
