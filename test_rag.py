#!/usr/bin/env python3
"""
Test script for the enhanced Persian RAG system.

This script demonstrates how to use the PersianRAG class to process documents,
create a vector store, and answer questions based on the processed content.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from rag_enhanced import PersianRAG, DocumentProcessor, VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_document_processor(data_dir: str):
    """Test the document processing pipeline."""
    print("\n" + "="*60)
    print("Testing Document Processor")
    print("="*60)
    
    processor = DocumentProcessor()
    
    # Test processing a directory
    print(f"\nProcessing documents in: {data_dir}")
    texts = processor.process_directory(data_dir)
    
    print(f"\nProcessed {len(texts)} text chunks")
    if texts:
        print("\nSample processed text:")
        print("-" * 50)
        print(texts[0][:500] + "..." if len(texts[0]) > 500 else texts[0])
        print("-" * 50)
    
    return texts

def test_vector_store(texts: List[str], save_path: str = "test_faiss_index"):
    """Test the vector store creation and loading."""
    print("\n" + "="*60)
    print("Testing Vector Store")
    print("="*60)
    
    if not texts:
        print("No texts provided for vector store testing")
        return None
    
    # Test vector store creation
    print("\nCreating vector store...")
    vsm = VectorStoreManager()
    
    if vsm.create_vector_store(texts, save_path):
        print(f"Vector store created successfully at: {save_path}")
        
        # Test loading the vector store
        print("\nTesting vector store loading...")
        vsm2 = VectorStoreManager()
        if vsm2.load_vector_store(save_path):
            print("Vector store loaded successfully!")
            return vsm2.vector_store
    
    print("Failed to create or load vector store")
    return None

def test_rag_system(data_dir: str, model_name: str = "qwen2.5"):
    """Test the complete RAG system."""
    print("\n" + "="*60)
    print("Testing RAG System")
    print("="*60)
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = PersianRAG(model_name=model_name)
    
    # Process documents
    print(f"\nProcessing documents in: {data_dir}")
    if not rag.process_documents(data_dir):
        print("Failed to process documents")
        return
    
    # Initialize QA chain
    print("\nInitializing QA chain...")
    if not rag.initialize_qa_chain():
        print("Failed to initialize QA chain")
        return
    
    # Test questions
    test_questions = [
        "چیستی هوش مصنوعی چیست؟",
        "تاریخچه هوش مصنوعی را توضیح دهید.",
        "کاربردهای هوش مصنوعی در زندگی روزمره چیست؟"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}\nسوال: {question}")
        print("-" * 60)
        result = rag.query(question)
        
        if 'error' in result:
            print(f"خطا: {result['error']}")
        else:
            print(f"\nپاسخ: {result['answer']}")
            
            if result.get('sources'):
                print("\nمنابع:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source.get('source', 'منبع ناشناخته')}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Persian RAG System')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--model', type=str, default='qwen2.5',
                       help='Ollama model name to use')
    parser.add_argument('--test', choices=['all', 'processor', 'vector', 'rag'], 
                       default='all', help='Which component to test')
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist")
        return
    
    try:
        if args.test in ['all', 'processor']:
            texts = test_document_processor(args.data_dir)
        else:
            texts = None
        
        if args.test in ['all', 'vector'] and texts:
            test_vector_store(texts)
        
        if args.test in ['all', 'rag']:
            test_rag_system(args.data_dir, args.model)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logger.exception("Error during testing")
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()
