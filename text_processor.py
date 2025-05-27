"""
Persian Text Processing Module

This module provides comprehensive text cleaning and processing utilities specifically
designed for Persian text, including normalization, cleaning, and preprocessing
for NLP tasks.
"""

import re
import string
import unicodedata
import html
import logging
from typing import List, Dict, Set, Optional, Callable, Any
from pathlib import Path
import json
import zlib
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PersianTextProcessor:
    """
    A comprehensive text processing class for Persian text with utilities for
    normalization, cleaning, and preprocessing.
    """
    
    # Persian stopwords
    STOPWORDS = {
        'و', 'در', 'به', 'از', 'که', 'با', 'را', 'این', 'برای', 'آن',
        'یک', 'خود', 'تا', 'کرد', 'شده', 'است', 'نیز', 'شود', 'های', 'اما',
        'کردن', 'کردم', 'کرده', 'کردی', 'کردید', 'کردند', 'خواهد', 'خواهیم',
        'خواهید', 'خواهند', 'شده', 'شده‌ام', 'شده‌ای', 'شده‌اید', 'شده‌اند',
        'شده‌بودم', 'شده‌بودی', 'شده‌بود', 'شده‌بودیم', 'شده‌بودید', 'شده‌بودند',
        'دارد', 'دارم', 'داری', 'داریم', 'دارید', 'دارند', 'داشتم', 'داشتی',
        'داشت', 'داشتیم', 'داشتید', 'داشتند', 'خواهم', 'خواهی', 'خواهد', 'خواهیم',
        'خواهید', 'خواهند', 'باشم', 'باشی', 'باشد', 'باشیم', 'باشید', 'باشند',
        'بودم', 'بودی', 'بود', 'بودیم', 'بودید', 'بودند', 'شو', 'شود', 'شویم',
        'شوید', 'شوند', 'شدم', 'شدی', 'شد', 'شدیم', 'شدید', 'شدند', 'م', 'ت', 'ش',
        'ای', 'ها', 'های', 'هایی', 'هایم', 'هایت', 'هایش', 'هایمان', 'هایتان', 'هایشان'
    }
    
    # Common Persian abbreviations and their normalized forms
    ABBREVIATIONS = {
        'دکتر': 'دکتر',
        'پروفسور': 'پروفسور',
        'آقای': 'آقای',
        'خانم': 'خانم',
        'استاد': 'استاد',
        'مهندس': 'مهندس',
    }
    
    # Common Persian prefixes and suffixes
    PREFIXES = ['می‌', 'نمی‌', 'بر', 'فرو', 'وا', 'پیش', 'هم', 'بی']
    SUFFIXES = ['ها', 'های', 'هایم', 'هایت', 'هایش', 'هایمان', 'هایتان', 'هایشان',
               'ان', 'ات', 'اش', 'ام', 'ای', 'تر', 'ترین', 'هایی']
    
    # Common noise patterns to remove
    NOISE_PATTERNS = [
        r'\d+',  # Numbers
        r'[a-zA-Z]',  # English letters
        r'[\[\](){}<>]',  # Brackets
        r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]',  # Punctuation
        r'\s+',  # Extra whitespace
        r'[\u200c\u200d\u200e\u200f\u202a-\u202e]'  # Special unicode
    ]
    
    def __init__(self, min_text_length: int = 20, max_text_length: int = 10000):
        """
        Initialize the Persian text processor.
        
        Args:
            min_text_length: Minimum length of text to keep after cleaning
            max_text_length: Maximum length of text to process
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self._compiled_patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+'),
            'email': re.compile(r'\S*@\S*\s?'),
            'hashtag': re.compile(r'#\S+'),
            'mention': re.compile(r'@\S+'),
        }
    
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Normalize Arabic characters to their Persian equivalents."""
        arabic_to_persian = {
            'ك': 'ک', 'ي': 'ی', 'ة': 'ه', 'ۀ': 'ه', 'أ': 'ا',
            'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ی', 'ً': '', 'ٌ': '',
            'ٍ': '', 'َ': '', 'ُ': '', 'ِ': '', 'ّ': '', 'ْ': '', 'ٰ': ''
        }
        for arabic, persian in arabic_to_persian.items():
            text = text.replace(arabic, persian)
        return text
    
    def clean_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Clean and normalize Persian text.
        
        Args:
            text: Input text to clean
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Skip very long texts
        if len(text) > self.max_text_length:
            logger.warning(f"Text too long ({len(text)} chars), skipping")
            return ""
        
        try:
            # Basic cleaning
            text = html.unescape(text)  # Decode HTML entities
            text = unicodedata.normalize('NFC', text)  # Normalize unicode
            text = self.normalize_arabic(text)  # Normalize Arabic to Persian
            
            # Remove URLs, emails, hashtags, mentions
            text = self._compiled_patterns['url'].sub(' ', text)
            text = self._compiled_patterns['email'].sub(' ', text)
            text = self._compiled_patterns['hashtag'].sub(' ', text)
            text = self._compiled_patterns['mention'].sub(' ', text)
            
            # Remove common noise patterns
            for pattern in self.NOISE_PATTERNS:
                text = re.sub(pattern, ' ', text)
            
            # Remove stopwords if requested
            if remove_stopwords:
                words = text.split()
                words = [word for word in words if word not in self.STOPWORDS]
                text = ' '.join(words)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            # Skip if text is too short after cleaning
            if len(text) < self.min_text_length:
                return ""
                
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""
    
    def process_document(self, text: str) -> Optional[str]:
        """
        Process a document and return cleaned text or None if too short.
        
        Args:
            text: Input document text
            
        Returns:
            Cleaned text or None if too short
        """
        cleaned = self.clean_text(text)
        return cleaned if len(cleaned) >= self.min_text_length else None
    
    def batch_process(self, texts: List[str], n_jobs: int = -1) -> List[str]:
        """
        Process a batch of texts in parallel.
        
        Args:
            texts: List of input texts
            n_jobs: Number of parallel jobs (-1 for all available cores)
            
        Returns:
            List of cleaned texts (empty strings for skipped texts)
        """
        from tqdm import tqdm
        
        processed = []
        for text in tqdm(texts, desc="Processing texts"):
            processed.append(self.clean_text(text))
            
        return processed
    
    @staticmethod
    def calculate_text_hash(text: str) -> str:
        """
        Calculate a hash for text to detect duplicates.
        
        Args:
            text: Input text
            
        Returns:
            MD5 hash of the normalized text
        """
        # Normalize text before hashing
        normalized = ' '.join(text.split()).strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts while preserving order.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of unique texts in original order
        """
        seen = set()
        unique_texts = []
        
        for text in texts:
            text_hash = self.calculate_text_hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)
                
        return unique_texts


# Example usage
if __name__ == "__main__":
    processor = PersianTextProcessor()
    
    # Example text
    sample_text = """
    این یک متن نمونه است با برخی از کلمات تکراری. 
    همچنین شامل برخی کاراکترهای عربی مانند: كلمة وعربی
    و لینک به سایت https://example.com و ایمیل test@example.com
    """
    
    cleaned = processor.clean_text(sample_text)
    print("Original:", sample_text)
    print("\nCleaned:", cleaned)
