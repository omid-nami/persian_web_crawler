#!/usr/bin/env python3
"""
Test script for the enhanced Persian RAG system.

This script demonstrates how to use the enhanced RAG system with sample data.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from rag_enhanced import PersianRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_enhanced_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_enhanced_rag():
    """Test the enhanced RAG system with sample data."""
    print("\n" + "="*60)
    print("Testing Enhanced Persian RAG System")
    print("="*60 + "\n")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = PersianRAG(model_name="qwen2.5")
    
    # Process sample data
    data_dir = "sample_data"
    print(f"\nProcessing sample data in: {data_dir}")
    
    # Create sample data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a sample CSV file if it doesn't exist
    sample_csv = os.path.join(data_dir, "sample_data.csv")
    if not os.path.exists(sample_csv):
        print("Sample data not found. Creating sample data...")
        from sample_data import create_sample_csv
        create_sample_csv(data_dir)
    
    # Process documents and create vector store
    print("\nProcessing documents and creating vector store...")
    if not rag.process_documents(data_dir):
        print("Failed to process documents")
        return
    
    # Initialize QA chain
    print("\nInitializing QA chain...")
    if not rag.initialize_qa_chain(k=3):
        print("Failed to initialize QA chain")
        return
    
    # Test questions
    test_questions = [
        "هوش مصنوعی چیست؟",
        "تاریخچه هوش مصنوعی را توضیح دهید.",
        "کاربردهای هوش مصنوعی چیست؟",
        "چالش‌های هوش مصنوعی چیست؟",
        "آینده هوش مصنوعی چگونه خواهد بود؟"
    ]
    
    print("\n" + "="*60)
    print("Testing RAG System with Sample Questions")
    print("="*60 + "\n")
    
    for question in test_questions:
        print(f"\n{'='*60}\nسوال: {question}")
        print("-" * 60)
        
        try:
            result = rag.query(question)
            
            if 'error' in result:
                print(f"خطا: {result['error']}")
            else:
                print(f"\nپاسخ: {result['answer']}")
                
                if result.get('sources'):
                    print("\nمنابع:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source.get('source', 'منبع ناشناخته')}")
                        
        except Exception as e:
            print(f"خطا در پردازش سوال: {str(e)}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_enhanced_rag()
