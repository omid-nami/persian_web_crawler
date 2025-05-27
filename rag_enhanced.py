"""
Enhanced RAG System for Persian Text

This module implements a robust Retrieval-Augmented Generation system
specifically designed for Persian text processing.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Local imports
from text_processor import PersianTextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, cleaning, and chunking."""
    
    def __init__(self, min_text_length: int = 20, max_text_length: int = 10000):
        """Initialize the document processor."""
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.text_processor = PersianTextProcessor(
            min_text_length=min_text_length,
            max_text_length=max_text_length
        )
    
    def process_csv_file(self, file_path: str) -> List[str]:
        """
        Process a single CSV file and return cleaned text chunks.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of cleaned text chunks
        """
        try:
            # Read CSV with error handling
            df = pd.read_csv(
                file_path,
                on_bad_lines='skip',
                encoding='utf-8',
                engine='python',
                dtype=str
            )
            
            if 'text' not in df.columns:
                logger.warning(f"'text' column not found in {file_path}")
                return []
                
            # Process and clean texts
            cleaned_texts = []
            for text in df['text'].dropna():
                cleaned = self.text_processor.process_document(text)
                if cleaned:
                    cleaned_texts.append(cleaned)
            
            return cleaned_texts
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def process_directory(self, data_dir: str) -> List[str]:
        """
        Process all CSV files in a directory.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            List of cleaned and deduplicated text chunks
        """
        all_texts = []
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return []
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Process each file
        for file in tqdm(csv_files, desc="Processing files"):
            file_path = os.path.join(data_dir, file)
            texts = self.process_csv_file(file_path)
            all_texts.extend(texts)
            
            # Log progress
            if len(all_texts) % 100 == 0:
                logger.info(f"Collected {len(all_texts)} text chunks")
        
        # Remove duplicates while preserving order
        unique_texts = self.text_processor.deduplicate_texts(all_texts)
        
        logger.info(f"After deduplication: {len(unique_texts)} unique text chunks")
        return unique_texts


class VectorStoreManager:
    """Manages the creation and loading of vector stores."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model
        """
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        
    def initialize_embeddings(self) -> bool:
        """Initialize the embedding model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                    "show_progress_bar": True
                }
            )
            logger.info(f"Initialized embedding model: {self.embedding_model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            return False
    
    def create_vector_store(self, texts: List[str], save_path: str = "faiss_index") -> bool:
        """
        Create a FAISS vector store from texts.
        
        Args:
            texts: List of text chunks to index
            save_path: Directory to save the vector store
            
        Returns:
            True if successful, False otherwise
        """
        if not texts:
            logger.error("No texts provided to create vector store")
            return False
            
        if not self.embeddings:
            if not self.initialize_embeddings():
                return False
        
        try:
            logger.info("Creating text splitter...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
                separators=['\n\n', '\n', ' ', '']
            )
            
            # Split documents
            logger.info("Splitting documents into chunks...")
            chunks = text_splitter.create_documents(texts)
            
            if not chunks:
                logger.error("No chunks created from documents")
                return False
                
            logger.info(f"Creating FAISS index with {len(chunks)} chunks...")
            
            # Create and save vector store
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                distance_strategy="COSINE"
            )
            
            # Ensure save directory exists
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            
            # Save the vector store
            self.vector_store.save_local(save_path)
            logger.info(f"Vector store created and saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False
    
    def load_vector_store(self, load_path: str = "faiss_index") -> bool:
        """
        Load a FAISS vector store from disk.
        
        Args:
            load_path: Directory containing the vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.embeddings:
                if not self.initialize_embeddings():
                    return False
            
            logger.info(f"Loading vector store from {load_path}...")
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False


class PersianRAG:
    """
    A Retrieval-Augmented Generation system for Persian text.
    """
    
    def __init__(self, model_name: str = "qwen2.5"):
        """
        Initialize the RAG system.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.vector_store = None
        self.qa_chain = None
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        
        # Initialize LLM with streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        logger.info(f"Initialized Persian RAG system with model: {model_name}")
    
    def process_documents(self, data_dir: str) -> bool:
        """
        Process documents and create a vector store.
        
        Args:
            data_dir: Directory containing document files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process documents
            logger.info(f"Processing documents in {data_dir}...")
            texts = self.document_processor.process_directory(data_dir)
            
            if not texts:
                logger.error("No valid text found in documents")
                return False
            
            # Create vector store
            return self.vector_store_manager.create_vector_store(texts)
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def load_vector_store(self, path: str = "faiss_index") -> bool:
        """
        Load a pre-built vector store.
        
        Args:
            path: Path to the vector store directory
            
        Returns:
            True if successful, False otherwise
        """
        return self.vector_store_manager.load_vector_store(path)
    
    def initialize_qa_chain(self, k: int = 4) -> bool:
        """
        Initialize the QA chain with the loaded vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self.vector_store_manager, 'vector_store') or \
               self.vector_store_manager.vector_store is None:
                logger.error("No vector store loaded")
                return False
            
            # Initialize LLM
            llm = Ollama(
                model=self.model_name,
                callback_manager=self.callback_manager,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                num_ctx=4096
            )
            
            # Create retriever
            retriever = self.vector_store_manager.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # Define prompt template
            template = """
            شما یک دستیار هوشمند هستید که به سوالات بر اساس متن‌های داده شده پاسخ می‌دهید.
            
            متن‌های مرتبط:
            {context}
            
            سوال: {question}
            پاسخ:
            """
            
            QA_PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )
            
            logger.info("QA chain initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if not self.qa_chain:
            return {"error": "QA chain not initialized. Call initialize_qa_chain() first."}
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.metadata for doc in result["source_documents"]]
            }
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {"error": f"An error occurred: {str(e)}"}


def main():
    """Main function to demonstrate the RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persian RAG System")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                       help="Directory containing processed data")
    parser.add_argument("--model", type=str, default="qwen2.5",
                       help="Ollama model name")
    parser.add_argument("--load-vector-store", action="store_true",
                       help="Load existing vector store instead of creating new one")
    parser.add_argument("--vector-store-path", type=str, default="faiss_index",
                       help="Path to save/load the vector store")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = PersianRAG(model_name=args.model)
    
    # Process documents or load existing vector store
    if args.load_vector_store:
        if not rag.load_vector_store(args.vector_store_path):
            logger.error("Failed to load vector store")
            return
    else:
        if not rag.process_documents(args.data_dir):
            logger.error("Failed to process documents")
            return
    
    # Initialize QA chain
    if not rag.initialize_qa_chain():
        logger.error("Failed to initialize QA chain")
        return
    
    # Interactive query loop
    print("\nPersian RAG System - Ready!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type '/sources' to see the sources for the last answer.\n")
    
    while True:
        try:
            question = input("\nسوال خود را بپرسید: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                print("\nخدانگهدار!")
                break
                
            if question.lower() == '/sources':
                if hasattr(rag, 'last_sources') and rag.last_sources:
                    print("\nمنابع پاسخ آخر:")
                    for i, source in enumerate(rag.last_sources, 1):
                        print(f"{i}. {source.get('source', 'Unknown')}")
                else:
                    print("\nمنبعی برای نمایش وجود ندارد.")
                continue
                
            print("\nدر حال پردازش سوال...")
            result = rag.query(question)
            
            if 'error' in result:
                print(f"\nخطا: {result['error']}")
            else:
                print(f"\nپاسخ: {result['answer']}")
                rag.last_sources = result.get('sources', [])
                
                if rag.last_sources:
                    print("\n(برای مشاهده منابع از دستور /sources استفاده کنید)")
                
        except KeyboardInterrupt:
            print("\n\nبرنامه توسط کاربر متوقف شد.")
            break
        except Exception as e:
            print(f"\nخطای غیرمنتظره: {str(e)}")
            logger.exception("Unexpected error in main loop")


if __name__ == "__main__":
    main()
