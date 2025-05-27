#!/usr/bin/env python3
"""
Run the enhanced Persian RAG system with a user-friendly interface.

This script provides a simple command-line interface to interact with
the enhanced Persian RAG system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import local modules
try:
    from rag_enhanced import PersianRAG
except ImportError as e:
    print(f"Error importing RAG module: {e}")
    print("Please make sure you have installed all required dependencies.")
    print("You can install them using: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_interface.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print a fancy banner for the RAG interface."""
    banner = """
    [36m╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║       [1m[35mPersian RAG System - Enhanced Edition[0m[36m        ║
    ║       [33mRetrieval-Augmented Generation for Persian[36m      ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝[0m
    """
    print(banner)

def print_help():
    """Print help information for the interactive mode."""
    help_text = """
    [1mAvailable commands:[0m
    [32mhelp[0m, [32m?[0m      - Show this help message
    [32mclear[0m, [32mcls[0m    - Clear the screen
    [32msources[0m      - Show sources for the last answer
    [32mexit[0m, [32mquit[0m   - Exit the program
    [32mabout[0m       - Show information about the system
    
    [1mAsk a question[0m to get an answer from the RAG system.
    Example: [33mهوش مصنوعی چیست؟[0m
    """
    print(help_text)

def print_about():
    """Print information about the RAG system."""
    about_text = """
    [1mPersian RAG System - Enhanced Edition[0m
    Version: 1.0.0
    
    A Retrieval-Augmented Generation system for Persian text,
    built with LangChain and Ollama.
    
    Features:
    - Advanced Persian text processing
    - Efficient document retrieval using FAISS
    - Integration with Ollama LLM
    - Support for multiple document formats
    - Interactive query interface
    
    [33mNote:[0m Make sure the Ollama server is running and
    the desired model (e.g., qwen2.5) is downloaded.
    """
    print(about_text)

def interactive_mode(rag, data_dir):
    """Run the RAG system in interactive mode."""
    print("\n[32mEntering interactive mode. Type 'help' for commands.\n[0m")
    
    last_sources = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n[36m❯❯❯ [0m").strip()
            
            # Handle commands
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'خروج']:
                print("\n[33mخدانگهدار! (Goodbye!)[0m")
                break
                
            if user_input.lower() in ['help', '?', 'راهنما']:
                print_help()
                continue
                
            if user_input.lower() in ['clear', 'cls', 'پاک کردن']:
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
                
            if user_input.lower() in ['sources', 'منابع']:
                if last_sources:
                    print("\n[1mSources for the last answer:[0m")
                    for i, source in enumerate(last_sources, 1):
                        print(f"  {i}. {source.get('source', 'Unknown')}")
                        if 'content' in source:
                            print(f"     {source['content'][:150]}...")
                else:
                    print("\n[33mNo sources available for the last answer.[0m")
                continue
                
            if user_input.lower() in ['about', 'درباره']:
                print_about()
                continue
            
            # Process the query
            print("\n[33mدر حال پردازش سوال...[0m")
            result = rag.query(user_input)
            
            if 'error' in result:
                print(f"\n[31mخطا: {result['error']}[0m")
            else:
                # Print the answer
                print("\n[1m[34mپاسخ:[0m")
                print(f"{result['answer']}")
                
                # Store sources for later reference
                last_sources = result.get('sources', [])
                
                # Show sources if available
                if last_sources:
                    print("\n[1mمنابع:[0m")
                    for i, source in enumerate(last_sources, 1):
                        print(f"  {i}. {source.get('source', 'منبع ناشناخته')}")
        
        except KeyboardInterrupt:
            print("\n[33mبرای خروج از برنامه 'exit' یا 'quit' تایپ کنید.[0m")
            continue
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"\n[31mخطا در پردازش درخواست: {str(e)}[0m")

def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description='Persian RAG System - Enhanced Edition')
    
    # Add arguments
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Directory containing the data files (default: processed_data)')
    parser.add_argument('--model', type=str, default='qwen2.5',
                       help='Ollama model to use (default: qwen2.5)')
    parser.add_argument('--load-vector-store', action='store_true',
                       help='Load existing vector store instead of creating new one')
    parser.add_argument('--vector-store-path', type=str, default='faiss_index',
                       help='Path to the vector store directory (default: faiss_index)')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of document chunks to retrieve (default: 3)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Print banner
        print_banner()
        
        # Initialize RAG system
        print("\n[33mInitializing Persian RAG system...[0m")
        print(f"Model: {args.model}")
        print(f"Data directory: {args.data_dir}")
        
        rag = PersianRAG(model_name=args.model)
        
        # Process documents or load existing vector store
        if args.load_vector_store:
            print(f"\n[33mLoading vector store from {args.vector_store_path}...[0m")
            if not rag.load_vector_store(args.vector_store_path):
                print("\n[31mFailed to load vector store. Exiting...[0m")
                sys.exit(1)
        else:
            print(f"\n[33mProcessing documents in {args.data_dir}...[0m")
            if not rag.process_documents(args.data_dir):
                print("\n[31mFailed to process documents. Exiting...[0m")
                sys.exit(1)
        
        # Initialize QA chain
        print("\n[33mInitializing QA chain...[0m")
        if not rag.initialize_qa_chain(k=args.k):
            print("\n[31mFailed to initialize QA chain. Exiting...[0m")
            sys.exit(1)
        
        # Enter interactive mode
        interactive_mode(rag, args.data_dir)
        
    except KeyboardInterrupt:
        print("\n\n[33mبرنامه توسط کاربر متوقف شد.[0m")
        sys.exit(0)
    except Exception as e:
        logger.exception("Error in main function")
        print(f"\n[31mخطای غیرمنتظره: {str(e)}[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
