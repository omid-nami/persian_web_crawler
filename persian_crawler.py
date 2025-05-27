import os
import sys
import time
import json
import logging
import re
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set, Tuple, Union, Any, Callable
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from hazm import Normalizer, word_tokenize, stopwords_list
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from persiantools import characters
import PyPDF2
from io import BytesIO
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords

# Optional OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not installed. Install with: pip install pytesseract pdf2image")

# Configure number of workers (use all available CPU cores by default)
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1) if multiprocessing.cpu_count() > 1 else 1

# Configure Tesseract parameters for better performance
TESSERACT_CONFIG = {
    'lang': 'fas+eng',  # Try both Persian and English
    'config': '--oem 3 --psm 6',  # LSTM OCR Engine, Assume a single uniform block of text
    'nice': 0,  # Lower priority to avoid system freezing
    'timeout': 60  # Timeout in seconds per page
}

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords
ENGLISH_STOPWORDS = set(stopwords.words('english'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PersianWebCrawler:
    def __init__(self, base_url: str, max_pages: int = 100):
        """
        Initialize the Persian web crawler.
        
        Args:
            base_url (str): The starting URL to crawl
            max_pages (int): Maximum number of pages to crawl (default: 100)
        """
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited_urls = set()
        self.pages_crawled = 0
        self.data = []
        
        # Initialize Persian text normalizer
        self.normalizer = Normalizer()
        self.persian_stopwords = set(stopwords_list())
        
        # Create output directories
        self.raw_data_dir = Path('raw_data')
        self.processed_data_dir = Path('processed_data')
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain and is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == self.domain
    
    def get_page_links(self, url: str) -> List[str]:
        """Extract all valid links from a webpage with comprehensive URL handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            
            # Add a small delay to be respectful to the server
            time.sleep(0.5)
            
            response = self.session.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.encoding = 'utf-8'  # Ensure proper encoding for Persian text
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            links = set()  # Use a set to avoid duplicates
            base_domain = urlparse(url).netloc
            
            # Look for links in various locations
            link_sources = [
                soup.find_all('a', href=True),  # All links
                soup.select('nav a[href]'),     # Navigation links
                soup.select('.menu a[href]'),   # Menu links
                soup.select('footer a[href]'),  # Footer links
                soup.select('.content a[href]'),# Content area links
                soup.select('article a[href]'), # Article links
                soup.select('.pagination a[href]')  # Pagination links
            ]
            
            for link_source in link_sources:
                for a_tag in link_source:
                    try:
                        href = a_tag.get('href', '').strip()
                        if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                            continue
                            
                        # Handle relative URLs and normalize
                        full_url = urljoin(url, href)
                        parsed = urlparse(full_url)
                        
                        # Clean URL - remove fragments and query parameters for deduplication
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
                        
                        # Skip if not from the same domain or already visited
                        if parsed.netloc != base_domain:
                            continue
                            
                        # Skip file extensions we don't want
                        if any(clean_url.lower().endswith(ext) for ext in [
                            '.jpg', '.jpeg', '.png', '.gif', '.pdf', 
                            '.doc', '.docx', '.xls', '.xlsx', '.ppt', 
                            '.pptx', '.zip', '.rar', '.7z', '.exe', 
                            '.mp3', '.mp4', '.avi', '.mov', '.wmv',
                            '.css', '.js', '.ico', '.svg', '.woff', '.woff2',
                            '.ttf', '.eot', '.otf'
                        ]):
                            continue
                            
                        # Skip common non-content paths
                        if any(skip in clean_url.lower() for skip in [
                            '/wp-admin/', '/wp-content/', '/wp-includes/',
                            '/feed/', '/rss', '/atom', '/tag/', '/author/'
                        ]):
                            continue
                            
                        # Add to our links if not already visited
                        if clean_url not in self.visited_urls:
                            links.add(clean_url)
                            
                    except Exception as e:
                        logger.debug(f"Error processing URL {href}: {str(e)}")
                        continue
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return []
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        try:
            # Check for Persian characters
            if re.search(r'[\u0600-\u06FF]', text):
                return 'fa'
            
            # Use langdetect for other languages
            lang = detect(text)
            return lang if lang in ['fa', 'en'] else 'en'  # Default to English if not Persian
            
        except (LangDetectException, Exception):
            return 'en'  # Default to English if detection fails
    
    def clean_text(self, text: str, language: str = 'fa') -> Tuple[str, List[str]]:
        """Clean and tokenize text based on language."""
        if language == 'fa':
            # Normalize Persian text
            text = self.normalizer.normalize(text)
            # Tokenize and clean Persian text
            tokens = word_tokenize(text)
            cleaned_tokens = [
                word for word in tokens 
                if word not in self.persian_stopwords and word.isalpha()
            ]
        else:
            # Clean and tokenize English text
            text = text.lower()
            # Remove special characters but keep letters and basic punctuation
            text = re.sub(r'[^a-z\s]', ' ', text)
            tokens = text.split()
            cleaned_tokens = [
                word for word in tokens 
                if word not in ENGLISH_STOPWORDS and len(word) > 2
            ]
        return text, cleaned_tokens
    
    def _process_single_page(self, args: tuple) -> tuple[int, str]:
        """Process a single page image with OCR."""
        i, image = args
        try:
            # Use Tesseract to do OCR on the image
            text = pytesseract.image_to_string(
                image,
                lang=TESSERACT_CONFIG['lang'],
                config=TESSERACT_CONFIG['config'],
                timeout=TESSERACT_CONFIG['timeout']
            )
            return (i, text.strip() if text.strip() else None)
        except Exception as e:
            logger.error(f"Error in OCR processing page {i+1}: {str(e)}")
            return (i, None)
    
    def extract_text_with_ocr(self, file_path: Path) -> Optional[str]:
        """Extract text from a PDF using OCR with parallel processing."""
        if not OCR_AVAILABLE:
            logger.warning("OCR dependencies not available. Install pytesseract and pdf2image.")
            return None
            
        try:
            logger.info(f"Attempting OCR on: {file_path}")
            
            # Convert PDF to images with parallel processing
            try:
                images = convert_from_path(
                    str(file_path),
                    dpi=300,  # Higher DPI for better OCR accuracy
                    grayscale=True,  # Convert to grayscale for better OCR
                    thread_count=NUM_WORKERS  # Use multiple threads for PDF conversion
                )
            except (PDFPageCountError, PDFSyntaxError) as e:
                logger.error(f"Error converting PDF to images: {str(e)}")
                return None
            
            if not images:
                logger.warning(f"No pages could be converted to images: {file_path}")
                return None
            
            # Process pages in parallel
            texts = [None] * len(images)
            
            # Use ThreadPoolExecutor for I/O bound tasks (OCR)
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Create a progress bar
                with tqdm(total=len(images), desc="OCR Processing", unit="page") as pbar:
                    # Submit all pages for processing
                    future_to_page = {
                        executor.submit(self._process_single_page, (i, img)): i 
                        for i, img in enumerate(images)
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_page):
                        try:
                            page_num, text = future.result()
                            if text:
                                texts[page_num] = text
                        except Exception as e:
                            logger.error(f"Error processing OCR result: {str(e)}")
                        finally:
                            pbar.update(1)
            
            # Combine all non-None texts in order
            combined_text = '\n\n'.join(filter(None, texts))
            return combined_text if combined_text else None
            
        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {str(e)}")
            return None
    
    def extract_pdf_content(self, file_path: Union[str, Path], use_ocr: bool = True) -> Optional[Dict]:
        """Extract text content from a local PDF file with language detection.
        
        Args:
            file_path: Path to the PDF file
            use_ocr: Whether to attempt OCR if direct text extraction fails
            
        Returns:
            Dictionary containing extracted content and metadata, or None if extraction failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"PDF file not found: {file_path}")
                return None
                
            logger.info(f"Processing PDF: {file_path}")
            extraction_method = 'direct'
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # First try direct text extraction
                text_parts = []
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                full_text = '\n\n'.join(text_parts)
                
                # If no text found and OCR is enabled, try OCR
                if (not full_text.strip() or len(full_text.strip()) < 100) and use_ocr and OCR_AVAILABLE:
                    ocr_text = self.extract_text_with_ocr(file_path)
                    if ocr_text and len(ocr_text) > 100:  # Only use OCR if we got a reasonable amount of text
                        full_text = ocr_text
                        extraction_method = 'ocr'
                        logger.info(f"Successfully extracted text using OCR from: {file_path}")
                
                if not full_text.strip():
                    logger.warning(f"No text content found in PDF (tried {'OCR and ' if use_ocr else ''}direct extraction): {file_path}")
                    return None
                
                # Detect language
                sample_text = full_text[:1000]  # Use first 1000 chars for detection
                language = self.detect_language(sample_text)
                
                # Clean and process text based on language
                processed_text, tokens = self.clean_text(full_text, language)
                
                # Get basic metadata
                metadata = {
                    'url': f"file://{file_path.absolute()}",
                    'title': file_path.stem,
                    'text': full_text,
                    'processed_text': processed_text,
                    'tokens': tokens,
                    'language': language,
                    'publish_date': None,
                    'source': str(file_path.absolute()),
                    'file_type': 'pdf',
                    'file_size': os.path.getsize(file_path),
                    'page_count': len(pdf_reader.pages),
                    'extraction_method': extraction_method,
                    'crawl_timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Try to extract creation date from PDF metadata
                try:
                    if pdf_reader.metadata and pdf_reader.metadata.creation_date:
                        if hasattr(pdf_reader.metadata.creation_date, 'isoformat'):
                            metadata['creation_date'] = pdf_reader.metadata.creation_date.isoformat()
                        else:
                            metadata['creation_date'] = str(pdf_reader.metadata.creation_date)
                    if pdf_reader.metadata and pdf_reader.metadata.author:
                        metadata['author'] = pdf_reader.metadata.author
                except Exception as e:
                    logger.warning(f"Could not extract metadata: {str(e)}")
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return None
                
    def process_directory(self, directory: str, file_extensions: List[str] = None):
        """Process all files in a directory with given extensions."""
        if file_extensions is None:
            file_extensions = ['.pdf']  # Default to PDF files
            
        directory = Path(directory)
        if not directory.is_dir():
            logger.error(f"Directory not found: {directory}")
            return
            
        logger.info(f"Processing directory: {directory}")
        
        # Find all matching files
        files_to_process = []
        for ext in file_extensions:
            files_to_process.extend(directory.glob(f'**/*{ext}'))
            
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process each file
        for file_path in tqdm(files_to_process, desc="Processing files"):
            if file_path.suffix.lower() == '.pdf':
                page_data = self.extract_pdf_content(file_path)
                if page_data:
                    self.save_data(page_data)
                    self.pages_crawled += 1
                    
        # Save any remaining data
        if self.data:
            self._save_batch()
        
        logger.info(f"Completed processing {self.pages_crawled} files from {directory}")

    def extract_article_content(self, url: str) -> Optional[Dict]:
        """Extract and process article content with fallback mechanisms."""
        # Check if it's a file URL
        if url.startswith('file://'):
            file_path = url[7:]  # Remove 'file://' prefix
            if file_path.lower().endswith('.pdf'):
                return self.extract_pdf_content(file_path)
            
        # Define headers for the request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8'
        }
        
        # Process web URL
        try:
            # First try with newspaper3k
            try:
                article = Article(url, language='fa', keep_article_html=True, request_timeout=10)
                article.download()
                article.parse()
                
                if not article.text.strip():
                    raise ValueError("No content extracted by newspaper3k")
                    
                title = article.title
                text = article.text
                publish_date = str(article.publish_date) if article.publish_date else None
                
            except Exception as e:
                logger.warning(f"Newspaper3k extraction failed for {url}, falling back to direct extraction: {str(e)}")
                # Fallback to direct extraction
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Try to find title
                title_tag = soup.find('title')
                title = title_tag.text.strip() if title_tag else 'No title'
                
                # Try to find main content - this is site-specific and might need adjustment
                main_content = soup.find('article') or soup.find('div', class_=lambda x: x and 'content' in x.lower())
                if not main_content:
                    # If no specific content div found, use the body
                    main_content = soup.find('body')
                
                text = main_content.get_text(separator='\n', strip=True) if main_content else ''
                publish_date = None
            
            # Clean and normalize Persian text
            normalized_text = self.normalizer.normalize(text)
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(normalized_text)
            cleaned_tokens = [word for word in tokens if word not in self.persian_stopwords and word.isalpha()]
            
            return {
                'url': url,
                'title': title,
                'text': text,
                'normalized_text': normalized_text,
                'tokens': cleaned_tokens,
                'publish_date': publish_date,
                'source': urlparse(url).netloc,
                'crawl_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None
    
    def save_data(self, data: Dict, batch_size: int = 10):
        """Save crawled data in batches."""
        if not data:
            return
            
        self.data.append(data)
        
        # Save data in batches to prevent memory issues
        if len(self.data) >= batch_size:
            self._save_batch()
    
    def _save_batch(self):
        """Save the current batch of data to disk."""
        if not self.data:
            return
            
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save raw data as JSON
            raw_file = self.raw_data_dir / f'raw_data_{timestamp}.json'
            with open(raw_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            # Convert to DataFrame and clean data
            df = pd.DataFrame(self.data)
            
            # Save as CSV (more universally readable)
            csv_file = self.processed_data_dir / f'processed_data_{timestamp}.csv'
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            # Try to save as Parquet if possible
            try:
                parquet_file = self.processed_data_dir / f'processed_data_{timestamp}.parquet'
                df.to_parquet(parquet_file, index=False)
                logger.info(f"Saved batch of {len(self.data)} pages to {raw_file}, {csv_file}, and {parquet_file}")
            except Exception as e:
                logger.warning(f"Could not save as Parquet, using CSV only: {str(e)}")
                logger.info(f"Saved batch of {len(self.data)} pages to {raw_file} and {csv_file}")
                
        except Exception as e:
            logger.error(f"Error saving batch: {str(e)}")
            # Try to save at least the raw data
            try:
                raw_file = self.raw_data_dir / f'raw_data_error_{timestamp}.json'
                with open(raw_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved raw data to {raw_file} after error")
            except Exception as e2:
                logger.error(f"Failed to save raw data after error: {str(e2)}")
        finally:
            self.data = []  # Clear the batch
    
    def crawl(self):
        """Start crawling the website with comprehensive page discovery."""
        logger.info(f"Starting comprehensive crawl of {self.base_url}")
        
        # Initialize variables
        urls_to_visit = [self.base_url]
        self.visited_urls = set()  # Reset visited URLs
        self.pages_crawled = 0     # Reset page counter
        self.data = []             # Reset data
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Create a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })
        
        with tqdm(desc="Crawling pages", unit="page") as pbar:
            while urls_to_visit and consecutive_errors < max_consecutive_errors:
                current_url = urls_to_visit.pop(0)
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(current_url)
                
                try:
                    logger.info(f"Processing ({self.pages_crawled}): {current_url}")
                    
                    # Extract and process page content
                    page_data = self.extract_article_content(current_url)
                    
                    # Extract links from the page
                    links = self.get_page_links(current_url)
                    
                    # Add new links to the queue
                    for link in links:
                        if link not in urls_to_visit and link not in self.visited_urls:
                            urls_to_visit.append(link)
                    
                    # Save the page data
                    self.save_data(page_data)
                    
                    # Increment the page counter
                    self.pages_crawled += 1
                    
                    # Update the progress bar
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing {current_url}: {str(e)}")
                    consecutive_errors += 1
        
        # Save any remaining data
        self._save_batch()
        logger.info(f"Crawling completed. Crawled {self.pages_crawled} pages.")

def main():
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Persian Web Crawler')
    parser.add_argument('url', help='Base URL to start crawling from')
    parser.add_argument('--max-pages', type=int, default=100, 
                       help='Maximum number of pages to crawl (default: 100)')
    
    args = parser.parse_args()
    
    crawler = PersianWebCrawler(base_url=args.url, max_pages=args.max_pages)
    crawler.crawl()

if __name__ == "__main__":
    main()
